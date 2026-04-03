"""
Cross-Camera ReID — Single-file, fully self-contained
Changes in this version:
  • Full-resolution recordings (native camera res, not downscaled)
  • PRE_BUF_SEC=3 / POST_BUF_SEC=5 → proper 1-2 second clips
  • Face detection + face ID via insightface (buffalo_l model)
    Each tracked person gets:  body G-XXXX  +  face F-XXXX label
  • Face gallery stored alongside body gallery in ReIDEngine

Install:
    pip install ultralytics faiss-cpu flask flask-socketio \
                torch torchvision opencv-python requests \
                insightface onnxruntime

Usage:
    python3 main.py \
      --cam_a "rtsp://user:pass@IP:554/stream" \
      --cam_b "rtsp://user:pass@IP:554/stream" \
      --port  5000

Optional:
      --webhook "https://hooks.slack.com/services/..."
      --sound   "alert.wav"
      --model   yolov8s.pt
      --sim     0.75
      --conf    0.40
      --face_sim 0.45
"""

import argparse
import json
import logging
import os
import platform
import queue
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import faiss
import numpy as np
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO
from ultralytics import YOLO

# ── Optional: ResNet-50 body embedding ───────────────────────────────────────
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    _resnet.fc = torch.nn.Identity()
    _resnet.eval().to(_device)
    _tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 64)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    USE_RESNET = True
    BODY_EMB_DIM = 2048
    print(f"[reid] ResNet-50 body embeddings on {_device.upper()}")
except Exception:
    USE_RESNET = False
    BODY_EMB_DIM = 512
    print("[reid] Fallback: HSV color histogram (512-d)")

# ── Optional: insightface for face detection + embedding ─────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis

    import warnings, onnxruntime as _ort
    _avail_provs = _ort.get_available_providers()
    _providers   = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if p in _avail_provs]
    _face_app = FaceAnalysis(name="buffalo_l", providers=_providers)

    # insightface prints "Applied providers / find model" directly to C stdout
    # bypassing Python warnings — silence with fd-level redirect
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    _saved_fd1  = os.dup(1)
    _saved_fd2  = os.dup(2)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _face_app.prepare(ctx_id=0, det_size=(320, 320))
    finally:
        os.dup2(_saved_fd1, 1)
        os.dup2(_saved_fd2, 2)
        os.close(_devnull_fd)
        os.close(_saved_fd1)
        os.close(_saved_fd2)

    USE_FACE     = True
    FACE_EMB_DIM = 512
    _prov_label  = "GPU" if "CUDAExecutionProvider" in _providers else "CPU"
    print(f"[face] insightface buffalo_l loaded on {_prov_label}")
except Exception as e:
    USE_FACE = False
    FACE_EMB_DIM = 512
    print(f"[face] insightface not available ({e}) — face ID disabled")
    print("[face] Install: pip install insightface onnxruntime")

# ── Optional: sound + webhook ─────────────────────────────────────────────────
try:
    from playsound import playsound
    HAS_SOUND = True
except Exception:
    HAS_SOUND = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PERSON_CLASS    = 0
CONF_THRESH     = 0.40
SIM_THRESH      = 0.75
FACE_SIM_THRESH = 0.45
MAX_AGE_SEC     = 90
GALLERY_K       = 8
INFERENCE_SKIP  = 2
ALERT_COOLDOWN  = 10

DISPLAY_W = 960
DISPLAY_H = 540

PRE_BUF_SEC  = 3     # seconds of frames before detection (clip lead-in)
POST_BUF_SEC = 5     # seconds after last detection (clip tail)
CLIP_FPS     = 25    # output clip FPS

CAMERA_TOPOLOGY = {"cam_a": {"cam_b"}, "cam_b": {"cam_a"}}

_BODY_COLORS = [
    (52, 152, 219), (46, 204, 113), (231, 76, 60),  (155, 89, 182),
    (230, 126, 34), (26, 188, 156), (241, 196, 15), (236, 72, 153),
    (99, 110, 250), (52, 73,  94),
]

# ══════════════════════════════════════════════════════════════════════════════
# BODY EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
def embed_body(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros(BODY_EMB_DIM, np.float32)
    if USE_RESNET:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = _tfm(rgb).unsqueeze(0).to(_device)
        with torch.no_grad():
            v = _resnet(t).squeeze(0).cpu().numpy().astype(np.float32)
    else:
        r = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
        h  = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        s  = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        vc = cv2.calcHist([hsv], [2], None, [76],  [0, 256]).flatten()
        v  = np.concatenate([h, s, vc]).astype(np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# FACE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class FaceEngine:
    def __init__(self):
        self.lock     = threading.Lock()
        self.index    = faiss.IndexFlatIP(FACE_EMB_DIM)
        self.meta     = []
        self.gallery  = {}
        self.next_fid = 0

    def detect_and_embed(self, frame_bgr: np.ndarray, bbox: tuple):
        if not USE_FACE:
            return None, None
        x1, y1, x2, y2 = bbox
        face_y2   = y1 + max(1, (y2 - y1) // 2)
        face_crop = frame_bgr[max(0, y1):face_y2, max(0, x1):x2]
        if face_crop.size == 0:
            return None, None
        try:
            with self.lock:
                faces = _face_app.get(face_crop)
        except Exception:
            return None, None
        if not faces:
            return None, None
        face = max(faces, key=lambda f: f.det_score)
        if face.det_score < 0.5:
            return None, None
        emb = face.normed_embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        return emb, (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)

    def assign(self, face_emb: np.ndarray, ts: float) -> int:
        with self.lock:
            if self.index.ntotal == 0:
                return self._new(face_emb, ts)
            k = min(5, self.index.ntotal)
            sims, idxs = self.index.search(face_emb.reshape(1, -1), k)
            best_sim, best_fid = -1.0, None
            for sim, idx in zip(sims[0], idxs[0]):
                if idx < 0:
                    continue
                fid, src_ts = self.meta[idx]
                if ts - src_ts > MAX_AGE_SEC * 2:
                    continue
                if sim > best_sim:
                    best_sim, best_fid = sim, fid
            if best_sim >= FACE_SIM_THRESH and best_fid is not None:
                self._update(best_fid, face_emb, ts)
                return best_fid
            return self._new(face_emb, ts)

    def _new(self, emb, ts):
        fid = self.next_fid
        self.next_fid += 1
        self.gallery[fid] = [emb]
        self._add_idx(emb, fid, ts)
        return fid

    def _update(self, fid, emb, ts):
        g = self.gallery[fid]
        g.append(emb)
        if len(g) > GALLERY_K:
            g.pop(0)
        mean = np.stack(g).mean(0)
        mean /= (np.linalg.norm(mean) + 1e-6)
        self._add_idx(mean, fid, ts)

    def _add_idx(self, vec, fid, ts):
        self.index.add(vec.reshape(1, -1).astype(np.float32))
        self.meta.append((fid, ts))


# ══════════════════════════════════════════════════════════════════════════════
# ReID ENGINE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Person:
    gid:        int
    face_id:    int  = -1
    gallery:    list = field(default_factory=list)
    last_cam:   str  = ""
    last_seen:  float = 0.0
    trajectory: list = field(default_factory=list)


class ReIDEngine:
    def __init__(self):
        self.lock     = threading.Lock()
        self.index    = faiss.IndexFlatIP(BODY_EMB_DIM)
        self.meta     = []
        self.persons  = {}
        self.next_gid = 0
        self._l2g     = defaultdict(dict)

    def assign(self, cam_id, local_id, emb, ts):
        with self.lock:
            if local_id in self._l2g[cam_id]:
                gid = self._l2g[cam_id][local_id]
                self._update(gid, cam_id, emb, ts)
                return gid, False, None
            gid = self._search(cam_id, emb, ts)
            is_new   = gid is None
            from_cam = None
            if is_new:
                gid = self._new(cam_id, emb, ts)
            else:
                p = self.persons[gid]
                from_cam = p.last_cam if p.last_cam != cam_id else None
                self._update(gid, cam_id, emb, ts)
            self._l2g[cam_id][local_id] = gid
            return gid, is_new, from_cam

    def link_face(self, gid: int, face_id: int):
        with self.lock:
            if gid in self.persons and self.persons[gid].face_id == -1:
                self.persons[gid].face_id = face_id

    def _search(self, cam_id, emb, ts):
        if self.index.ntotal == 0:
            return None
        k = min(10, self.index.ntotal)
        sims, idxs = self.index.search(emb.reshape(1, -1), k)
        best_sim, best_gid = -1.0, None
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            gid, src_cam, src_ts = self.meta[idx]
            if ts - src_ts > MAX_AGE_SEC:
                continue
            if src_cam != cam_id:
                if cam_id not in CAMERA_TOPOLOGY.get(src_cam, set()):
                    continue
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        return best_gid if best_sim >= SIM_THRESH else None

    def _new(self, cam_id, emb, ts):
        gid = self.next_gid
        self.next_gid += 1
        p = Person(gid=gid, gallery=[emb], last_cam=cam_id,
                   last_seen=ts, trajectory=[cam_id])
        self.persons[gid] = p
        self._add_idx(emb, gid, cam_id, ts)
        return gid

    def _update(self, gid, cam_id, emb, ts):
        p = self.persons[gid]
        p.gallery.append(emb)
        if len(p.gallery) > GALLERY_K:
            p.gallery.pop(0)
        if cam_id != p.last_cam:
            p.trajectory.append(cam_id)
        p.last_cam  = cam_id
        p.last_seen = ts
        mean = np.stack(p.gallery).mean(0)
        mean /= (np.linalg.norm(mean) + 1e-6)
        self._add_idx(mean, gid, cam_id, ts)

    def _add_idx(self, vec, gid, cam_id, ts):
        self.index.add(vec.reshape(1, -1).astype(np.float32))
        self.meta.append((gid, cam_id, ts))

    def trajectory_report(self):
        return [
            {
                "global_id":    f"G-{p.gid:04d}",
                "face_id":      f"F-{p.face_id:04d}" if p.face_id >= 0 else "none",
                "trajectory":   p.trajectory,
                "cameras_seen": list(dict.fromkeys(p.trajectory)),
                "last_camera":  p.last_cam,
            }
            for p in sorted(self.persons.values(), key=lambda x: x.gid)
        ]


# ══════════════════════════════════════════════════════════════════════════════
# CAPTURE THREAD
# ══════════════════════════════════════════════════════════════════════════════
class CaptureThread(threading.Thread):
    def __init__(self, cam_id: str, url: str):
        super().__init__(daemon=True)
        self.cam_id    = cam_id
        self.url       = url
        self._frame    = None
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self.connected = False
        self.fps_src   = 0.0
        self.frame_w   = 1280
        self.frame_h   = 720

    def run(self):
        retry = 0
        while not self._stop.is_set():
            # Force TCP transport (more reliable on LAN than UDP)
            # and set a 10-second open timeout to fail fast on bad URLs
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "stimeout;10000000|"    # socket open timeout (microseconds)
                "timeout;10000000|"     # read timeout (microseconds)
                "err_detect;ignore_err|" # suppress H.264 MB decode errors
                "flags2;+fast|"         # fast error concealment
                "loglevel;quiet"        # suppress all FFmpeg console output
            )
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                retry += 1
                print(f"[{self.cam_id}] Connect failed (attempt {retry}), retry in 3s")
                time.sleep(3)
                continue
            self.connected = True
            print(f"[{self.cam_id}] RTSP connected")
            t0, fc = time.time(), 0
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.cam_id}] Stream lost, reconnecting")
                    self.connected = False
                    break
                h, w = frame.shape[:2]
                self.frame_w, self.frame_h = w, h
                with self._lock:
                    self._frame = frame
                fc += 1
                elapsed = time.time() - t0
                if elapsed >= 2.0:
                    self.fps_src = fc / elapsed
                    fc, t0 = 0, time.time()
            cap.release()

    def latest_frame(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class AlertEngine(threading.Thread):
    def __init__(self, bus, webhook_url=None, sound_path=None, log_path="alerts.log"):
        super().__init__(daemon=True)
        self.bus         = bus
        self.webhook_url = webhook_url
        self.sound_path  = sound_path
        self._cooldown   = {}
        self._stop       = threading.Event()
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format="%(asctime)s  %(message)s")

    def run(self):
        while not self._stop.is_set():
            try:
                ev = self.bus.get(timeout=0.5)
            except queue.Empty:
                continue
            if ev.get("type") not in ("handoff", "new_person"):
                self.bus.task_done()
                continue
            gid = ev["global_id"]
            now = ev.get("timestamp", time.time())
            if now - self._cooldown.get(gid, 0) < ALERT_COOLDOWN:
                self.bus.task_done()
                continue
            self._cooldown[gid] = now
            self._fire(ev)
            self.bus.task_done()

    def _fire(self, ev):
        gid      = ev["global_id"]
        fid      = ev.get("face_id", -1)
        ts_str   = datetime.fromtimestamp(ev.get("timestamp", time.time())).strftime("%H:%M:%S")
        face_str = f"  face=F-{fid:04d}" if fid >= 0 else ""
        if ev["type"] == "handoff":
            msg = (f"[HANDOFF] G-{gid:04d}{face_str}  "
                   f"{ev['from_cam'].upper()} -> {ev['to_cam'].upper()} @ {ts_str}")
        else:
            msg = (f"[NEW PERSON] G-{gid:04d}{face_str} "
                   f"on {ev.get('cam_id','?').upper()} @ {ts_str}")
        print(f"\033[93m{msg}\033[0m")
        logging.getLogger().info(msg)
        self._desktop(msg)
        if self.webhook_url and HAS_REQUESTS:
            try:
                _requests.post(self.webhook_url, json={"text": msg, **ev}, timeout=3)
            except Exception as e:
                print(f"[alert] Webhook error: {e}")
        if self.sound_path and HAS_SOUND:
            threading.Thread(target=playsound, args=(self.sound_path,), daemon=True).start()

    def _desktop(self, msg):
        try:
            sys = platform.system()
            if sys == "Linux":
                subprocess.Popen(["notify-send", "ReID Alert", msg],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys == "Darwin":
                subprocess.Popen(["osascript", "-e",
                    f'display notification "{msg}" with title "ReID Alert"'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
# RECORDER — native resolution, 3s lead-in + 5s tail = ~8s minimum clip
# ══════════════════════════════════════════════════════════════════════════════
class PersonRecorder(threading.Thread):
    def __init__(self, bus, out_dir="recordings"):
        super().__init__(daemon=True)
        self.bus     = bus
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._prebuf  = defaultdict(list)
        self._writers = {}              # key -> (VideoWriter, last_ts, (w, h))
        self._lock    = threading.Lock()
        self._stop    = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                ev = self.bus.get(timeout=0.3)
            except queue.Empty:
                self._close_stale()
                continue
            if ev.get("type") != "frame":
                self.bus.task_done()
                continue

            cam_id = ev["cam_id"]
            gid    = ev["global_id"]
            frame  = ev["frame"]        # full native-res annotated frame
            ts     = ev.get("timestamp", time.time())
            key    = (gid, cam_id)
            fh, fw = frame.shape[:2]

            with self._lock:
                buf = self._prebuf[key]
                buf.append((ts, frame.copy()))
                cutoff = ts - PRE_BUF_SEC
                while buf and buf[0][0] < cutoff:
                    buf.pop(0)
                if key not in self._writers:
                    self._open(key, gid, cam_id, ts, buf, fw, fh)
                else:
                    writer, _, dims = self._writers[key]
                    out_f = cv2.resize(frame, dims) if (fw, fh) != dims else frame
                    writer.write(out_f)
                    self._writers[key] = (writer, ts, dims)

            self.bus.task_done()
            self._close_stale()

    def _open(self, key, gid, cam_id, ts, buf, w, h):
        d  = self.out_dir / f"G-{gid:04d}"
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{cam_id}_{datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')}.mp4"
        wr = cv2.VideoWriter(str(fn), cv2.VideoWriter_fourcc(*"mp4v"),
                             CLIP_FPS, (w, h))
        for _, f in buf:
            wr.write(cv2.resize(f, (w, h)) if f.shape[1] != w else f)
        self._writers[key] = (wr, ts, (w, h))
        print(f"[recorder] New clip -> {fn}  ({w}x{h})")

    def _close_stale(self):
        now = time.time()
        with self._lock:
            stale = [k for k, (_, lt, _) in self._writers.items()
                     if now - lt > POST_BUF_SEC]
            for k in stale:
                self._writers.pop(k)[0].release()
                print(f"[recorder] Clip closed  G-{k[0]:04d}/{k[1]}")

    def stop(self):
        self._stop.set()
        with self._lock:
            for w, _, _ in self._writers.values():
                w.release()


# ══════════════════════════════════════════════════════════════════════════════
# FLASK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
_app = Flask(__name__)
_sio = SocketIO(_app, cors_allowed_origins="*", async_mode="threading")
_latest_frames = {}
_frame_lock    = threading.Lock()
_stats         = {"persons": 0, "handoffs": 0, "faces": 0, "events": []}
_stats_lock    = threading.Lock()

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-Camera ReID</title>
<script src="https://cdn.jsdelivr.net/npm/socket.io/dist/socket.io.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f1117;color:#e2e8f0;font-family:system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 20px;background:#1a1d27;border-bottom:1px solid #2d3148;flex-shrink:0}
header h1{font-size:15px;font-weight:500;color:#a5b4fc}
.dot{width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;margin-right:6px;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.statbar{display:flex;gap:28px;padding:10px 20px;background:#13151f;border-bottom:1px solid #2d3148;flex-shrink:0}
.stat{display:flex;flex-direction:column}
.sv{font-size:26px;font-weight:600;color:#a5b4fc}
.sl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em}
.body{display:grid;grid-template-columns:1fr 1fr 300px;gap:10px;padding:12px;flex:1;min-height:0}
.campanel{display:flex;flex-direction:column;gap:6px}
.camlbl{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em}
.campanel img{width:100%;border-radius:8px;border:1px solid #2d3148;background:#1a1d27;object-fit:contain}
.evpanel{display:flex;flex-direction:column;gap:6px;overflow:hidden}
.evpanel h2{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em}
#evlist{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:5px}
.ec{background:#1a1d27;border:1px solid #2d3148;border-radius:7px;padding:9px 11px;font-size:12px;animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(-5px)}to{opacity:1;transform:none}}
.ec.handoff{border-left:3px solid #f59e0b}
.ec.new_person{border-left:3px solid #22c55e}
.etop{display:flex;justify-content:space-between;margin-bottom:2px}
.egid{font-weight:600;color:#a5b4fc}
.etime{color:#475569;font-size:10px}
.edet{color:#94a3b8;font-size:11px}
.efid{color:#5eead4;font-size:10px;margin-top:2px}
.badge{display:inline-block;font-size:9px;padding:1px 5px;border-radius:4px;margin-right:4px;font-weight:700}
.bh{background:#451a03;color:#f59e0b}
.bn{background:#052e16;color:#22c55e}
#cs{font-size:11px;color:#475569}
</style>
</head>
<body>
<header>
  <h1><span class="dot"></span>Cross-Camera ReID</h1>
  <span id="cs">connecting...</span>
</header>
<div class="statbar">
  <div class="stat"><span class="sv" id="sp">0</span><span class="sl">Body IDs</span></div>
  <div class="stat"><span class="sv" id="sf">0</span><span class="sl">Face IDs</span></div>
  <div class="stat"><span class="sv" id="sh">0</span><span class="sl">Handoffs</span></div>
  <div class="stat"><span class="sv" id="st">--:--</span><span class="sl">Time</span></div>
</div>
<div class="body">
  <div class="campanel"><div class="camlbl">Camera A</div><img id="fa" src="/feed/cam_a"></div>
  <div class="campanel"><div class="camlbl">Camera B</div><img id="fb" src="/feed/cam_b"></div>
  <div class="evpanel"><h2>Live events</h2><div id="evlist"></div></div>
</div>
<script>
const s=io();
s.on("connect",()=>{document.getElementById("cs").textContent="connected";document.getElementById("cs").style.color="#22c55e"});
s.on("disconnect",()=>{document.getElementById("cs").textContent="disconnected";document.getElementById("cs").style.color="#ef4444"});
s.on("stats",d=>{
  document.getElementById("sp").textContent=d.persons;
  document.getElementById("sh").textContent=d.handoffs;
  document.getElementById("sf").textContent=d.faces;
});
s.on("event",ev=>{
  const l=document.getElementById("evlist"),c=document.createElement("div");
  c.className="ec "+ev.type;
  const faceRow=ev.face_id?`<div class="efid">face ${ev.face_id}</div>`:"";
  c.innerHTML=`<div class="etop"><span class="egid"><span class="badge ${ev.type==="handoff"?"bh":"bn"}">${ev.type==="handoff"?"HANDOFF":"NEW"}</span>${ev.global_id}</span><span class="etime">${ev.time}</span></div><div class="edet">${ev.detail}</div>${faceRow}`;
  l.prepend(c);while(l.children.length>50)l.removeChild(l.lastChild);
});
setInterval(()=>{document.getElementById("st").textContent=new Date().toLocaleTimeString()},1000);
</script>
</body></html>"""


def push_frame(cam_id: str, frame: np.ndarray):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
    with _frame_lock:
        _latest_frames[cam_id] = buf.tobytes()


def push_event(ev: dict):
    with _stats_lock:
        if ev["type"] == "handoff":
            _stats["handoffs"] += 1
        _stats["persons"] = ev.get("total_persons", _stats["persons"])
        _stats["faces"]   = ev.get("total_faces",   _stats["faces"])
        fid_str = f"F-{ev['face_id']:04d}" if ev.get("face_id", -1) >= 0 else None
        entry = {
            "time":      time.strftime("%H:%M:%S",
                         time.localtime(ev.get("timestamp", time.time()))),
            "type":      ev["type"],
            "global_id": f"G-{ev['global_id']:04d}",
            "face_id":   fid_str,
            "detail":    (f"{ev.get('from_cam','?').upper()} -> "
                          f"{ev.get('to_cam','?').upper()}")
                         if ev["type"] == "handoff"
                         else ev.get("cam_id", "?").upper(),
        }
        _stats["events"].insert(0, entry)
        _stats["events"] = _stats["events"][:50]
    _sio.emit("event", entry)
    _sio.emit("stats", {"persons": _stats["persons"],
                        "handoffs": _stats["handoffs"],
                        "faces":    _stats["faces"]})


def _mjpeg(cam_id):
    while True:
        with _frame_lock:
            jpg = _latest_frames.get(cam_id)
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.04)


@_app.route("/feed/<cam_id>")
def feed(cam_id):
    return Response(_mjpeg(cam_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@_app.route("/")
def index():
    return render_template_string(INDEX_HTML)


def run_dashboard(port=5000):
    print(f"[dashboard] http://0.0.0.0:{port}")
    _sio.run(_app, host="0.0.0.0", port=port,
             allow_unsafe_werkzeug=True, log_output=False)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE THREAD
# ══════════════════════════════════════════════════════════════════════════════
class InferenceThread(threading.Thread):
    def __init__(self, cam_id, capture, model,
                 engine: ReIDEngine, face_engine: FaceEngine,
                 alert_bus, record_bus):
        super().__init__(daemon=True)
        self.cam_id      = cam_id
        self.capture     = capture
        self.model       = model
        self.engine      = engine
        self.face_engine = face_engine
        self.alert_bus   = alert_bus
        self.record_bus  = record_bus
        self._stop       = threading.Event()
        self._skip       = 0
        self.fps         = 0.0

    def run(self):
        t0, fc = time.time(), 0
        while not self._stop.is_set():
            frame = self.capture.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            self._skip += 1
            annotated = frame.copy()
            ts = time.time()

            if self._skip >= INFERENCE_SKIP:
                self._skip = 0
                try:
                    results = self.model.track(
                        frame, persist=True,
                        classes=[PERSON_CLASS],
                        conf=CONF_THRESH,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"[{self.cam_id}] Inference error: {e}")
                    continue

                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        if box.id is None:
                            continue
                        lid = int(box.id.item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        crop = frame[max(0, y1):y2, max(0, x1):x2]

                        # Body ReID
                        body_emb = embed_body(crop)
                        gid, is_new, from_cam = self.engine.assign(
                            self.cam_id, lid, body_emb, ts)

                        # Face detection + face ID
                        face_emb, face_bbox = self.face_engine.detect_and_embed(
                            frame, (x1, y1, x2, y2))
                        face_id = -1
                        if face_emb is not None:
                            face_id = self.face_engine.assign(face_emb, ts)
                            self.engine.link_face(gid, face_id)

                        person     = self.engine.persons.get(gid)
                        linked_fid = person.face_id if person else -1

                        # Publish events
                        if from_cam and from_cam != self.cam_id:
                            self._publish({
                                "type":          "handoff",
                                "global_id":     gid,
                                "face_id":       linked_fid,
                                "from_cam":      from_cam,
                                "to_cam":        self.cam_id,
                                "timestamp":     ts,
                                "total_persons": len(self.engine.persons),
                                "total_faces":   self.face_engine.next_fid,
                            })
                        elif is_new:
                            self._publish({
                                "type":          "new_person",
                                "global_id":     gid,
                                "face_id":       linked_fid,
                                "cam_id":        self.cam_id,
                                "timestamp":     ts,
                                "total_persons": len(self.engine.persons),
                                "total_faces":   self.face_engine.next_fid,
                            })

                        # Draw body box + combined label
                        color = _BODY_COLORS[gid % len(_BODY_COLORS)]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"G-{gid:04d}"
                        if linked_fid >= 0:
                            label += f"  F-{linked_fid:04d}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(annotated,
                                      (x1, y1 - th - 10), (x1 + tw + 8, y1),
                                      color, -1)
                        cv2.putText(annotated, label, (x1 + 4, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 255, 255), 1)

                        # Draw face box (teal)
                        if face_bbox is not None:
                            fx1, fy1, fx2, fy2 = face_bbox
                            cv2.rectangle(annotated,
                                          (fx1, fy1), (fx2, fy2),
                                          (0, 220, 200), 2)
                            if face_id >= 0:
                                cv2.putText(annotated, f"F-{face_id:04d}",
                                            (fx1, max(0, fy1 - 4)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                            (0, 220, 200), 1)

                        # Send full-res annotated frame to recorder
                        try:
                            self.record_bus.put_nowait({
                                "type":      "frame",
                                "cam_id":    self.cam_id,
                                "global_id": gid,
                                "frame":     annotated.copy(),
                                "timestamp": ts,
                            })
                        except queue.Full:
                            pass

            # HUD
            st_txt   = "LIVE" if self.capture.connected else "RECONNECTING"
            st_color = (0, 210, 70) if self.capture.connected else (0, 80, 220)
            cv2.putText(annotated, f"{self.cam_id.upper()}  {st_txt}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, st_color, 2)
            cv2.putText(annotated,
                        f"src {self.capture.fps_src:.1f}fps  inf {self.fps:.1f}fps"
                        + ("  [face ON]" if USE_FACE else "  [face OFF]"),
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1)

            push_frame(self.cam_id,
                       cv2.resize(annotated, (DISPLAY_W, DISPLAY_H)))

            fc += 1
            elapsed = time.time() - t0
            if elapsed >= 2.0:
                self.fps = fc / elapsed
                fc, t0 = 0, time.time()

    def _publish(self, ev):
        try:
            self.alert_bus.put_nowait(ev)
        except queue.Full:
            pass
        push_event(ev)

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global CONF_THRESH, SIM_THRESH, FACE_SIM_THRESH

    ap = argparse.ArgumentParser(description="Cross-Camera ReID")
    ap.add_argument("--cam_a",    required=True)
    ap.add_argument("--cam_b",    required=True)
    ap.add_argument("--model",    default="yolov8n.pt")
    ap.add_argument("--conf",     type=float, default=CONF_THRESH)
    ap.add_argument("--sim",      type=float, default=SIM_THRESH)
    ap.add_argument("--face_sim", type=float, default=FACE_SIM_THRESH)
    ap.add_argument("--webhook",  default=None)
    ap.add_argument("--sound",    default=None)
    ap.add_argument("--port",     type=int, default=5000)
    args = ap.parse_args()

    CONF_THRESH     = args.conf
    SIM_THRESH      = args.sim
    FACE_SIM_THRESH = args.face_sim

    cameras    = {"cam_a": args.cam_a, "cam_b": args.cam_b}
    alert_bus  = queue.Queue(maxsize=200)
    record_bus = queue.Queue(maxsize=600)
    engine     = ReIDEngine()
    face_engine = FaceEngine()

    print(f"[main] Loading {args.model} ...")
    model = YOLO(args.model)

    AlertEngine(alert_bus,
                webhook_url=args.webhook,
                sound_path=args.sound).start()
    PersonRecorder(record_bus).start()

    threads = []
    for cam_id, url in cameras.items():
        cap = CaptureThread(cam_id, url)
        inf = InferenceThread(cam_id, cap, model, engine,
                              face_engine, alert_bus, record_bus)
        cap.start()
        inf.start()
        threads += [cap, inf]

    threading.Thread(target=run_dashboard,
                     kwargs={"port": args.port},
                     daemon=True).start()

    print(f"\n[main] Dashboard   -> http://localhost:{args.port}")
    print(f"[main] Face ID     -> {'enabled (insightface)' if USE_FACE else 'DISABLED  —  pip install insightface onnxruntime'}")
    print(f"[main] Recordings  -> recordings/G-XXXX/  (native resolution, ~{PRE_BUF_SEC+POST_BUF_SEC}s clips)")
    print("[main] Press Ctrl+C to quit\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[main] Shutting down ...")
        for t in threads:
            t.stop()
        out = Path("trajectory_final.json")
        out.write_text(json.dumps(engine.trajectory_report(), indent=2))
        print(f"[main] Trajectory -> {out}")
        print(f"[main] Recordings -> recordings/")
        print(f"[main] Alert log  -> alerts.log")


if __name__ == "__main__":
    main()