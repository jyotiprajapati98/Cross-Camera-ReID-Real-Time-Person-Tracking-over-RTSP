"""
Recorder — saves a short video clip for every person detection event.
Creates:  recordings/G-0042/cam_a_20240401_103045.mp4
"""

import threading
import queue
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

CLIP_PRE_BUFFER_SEC  = 2     # seconds of frames to keep before detection
CLIP_POST_BUFFER_SEC = 4     # seconds to keep recording after last detection
DISPLAY_FPS          = 25    # assumed FPS for VideoWriter
CLIP_W, CLIP_H       = 640, 360


class PersonRecorder(threading.Thread):
    """
    Consumes 'frame' events from event_bus and writes per-person clips.

    Frame event shape:
        {
          "type":       "frame",
          "cam_id":     "cam_a",
          "global_id":  42,
          "frame":      np.ndarray,    # annotated BGR frame
          "timestamp":  float,
        }
    """
    def __init__(self, event_bus: queue.Queue,
                 output_dir: str = "recordings"):
        super().__init__(daemon=True)
        self.bus        = event_bus
        self.out_dir    = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stop      = threading.Event()

        # gid+cam → list of (timestamp, frame) ring buffer
        self._pre_buf   = defaultdict(list)
        # gid+cam → (VideoWriter, last_activity_ts)
        self._writers   = {}
        self._lock      = threading.Lock()

    def run(self):
        while not self._stop.is_set():
            try:
                event = self.bus.get(timeout=0.3)
            except queue.Empty:
                self._close_stale_writers()
                continue

            if event.get("type") != "frame":
                self.bus.task_done()
                continue

            cam_id    = event["cam_id"]
            gid       = event["global_id"]
            frame     = event["frame"]
            ts        = event.get("timestamp", time.time())
            key       = (gid, cam_id)

            resized = cv2.resize(frame, (CLIP_W, CLIP_H))

            with self._lock:
                # Maintain pre-detection ring buffer
                buf = self._pre_buf[key]
                buf.append((ts, resized.copy()))
                # Trim to PRE_BUFFER duration
                cutoff = ts - CLIP_PRE_BUFFER_SEC
                while buf and buf[0][0] < cutoff:
                    buf.pop(0)

                # Open or refresh writer
                if key not in self._writers:
                    self._open_writer(key, gid, cam_id, ts, buf)
                else:
                    writer, _ = self._writers[key]
                    writer.write(resized)
                    self._writers[key] = (writer, ts)

            self.bus.task_done()
            self._close_stale_writers()

    def _open_writer(self, key, gid, cam_id, ts, pre_buf):
        clip_dir = self.out_dir / f"G-{gid:04d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        ts_str  = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        path    = clip_dir / f"{cam_id}_{ts_str}.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
        writer  = cv2.VideoWriter(str(path), fourcc, DISPLAY_FPS, (CLIP_W, CLIP_H))

        # Flush pre-buffer into clip
        for _, f in pre_buf:
            writer.write(f)

        self._writers[key] = (writer, ts)
        print(f"[recorder] New clip → {path}")

    def _close_stale_writers(self):
        now = time.time()
        with self._lock:
            to_close = [
                k for k, (_, last_ts) in self._writers.items()
                if now - last_ts > CLIP_POST_BUFFER_SEC
            ]
            for k in to_close:
                writer, _ = self._writers.pop(k)
                writer.release()
                print(f"[recorder] Clip closed for G-{k[0]:04d} / {k[1]}")

    def stop(self):
        self._stop.set()
        with self._lock:
            for writer, _ in self._writers.values():
                writer.release()
        self._writers.clear()