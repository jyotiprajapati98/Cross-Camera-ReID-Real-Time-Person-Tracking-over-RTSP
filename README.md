# Cross-Camera ReID — Real-Time Person Tracking over RTSP

A single-file, production-ready **Re-Identification (ReID)** system that tracks people across multiple IP cameras in real time. Detects persons with YOLOv8, identifies faces with InsightFace, matches identities across cameras using deep metric learning + FAISS vector search, and streams a live dashboard to your browser.

---

## Features

| Feature | Detail |
|---|---|
| **Person detection** | YOLOv8n pretrained on COCO — no training required |
| **Body ReID** | ResNet-50 (2048-d) or HSV histogram fallback |
| **Face ID** | InsightFace `buffalo_l` ArcFace (512-d) |
| **Cross-camera matching** | FAISS cosine ANN with temporal + topology gating |
| **Live dashboard** | Flask + SocketIO MJPEG stream at `http://localhost:5000` |
| **Alerts** | Terminal + desktop notification + Slack/Teams webhook + audio |
| **Recordings** | Per-person MP4 clips at native camera resolution (~8s per clip) |
| **Auto-reconnect** | RTSP streams reconnect automatically on dropout |
| **Single file** | Everything in `main.py` — no submodules |

---

## Architecture

```
RTSP Cam A ──► CaptureThread A ──► InferenceThread A ──┐
                                   (YOLO + ReID + Face)  │
                                                          ▼
                                                    Event Bus (Queue)
                                                          │
                                   ┌──────────────────────┼──────────────────┐
                                   ▼                      ▼                  ▼
                             AlertEngine           PersonRecorder      Flask Dashboard
                          (webhook/sound/log)    (MP4 clips/native)  (MJPEG + SocketIO)

RTSP Cam B ──► CaptureThread B ──► InferenceThread B ──► Shared ReIDEngine + FaceEngine
                                                          (FAISS index, thread-safe Lock)
```

Each camera runs two daemon threads:
- **CaptureThread** — continuously drains the RTSP buffer, always serves the freshest frame
- **InferenceThread** — runs YOLO tracking + body embedding + face detection every N frames

All threads share a single `ReIDEngine` and `FaceEngine` protected by `threading.Lock`.

---

## Requirements

### Python
- Python 3.10 or higher

### System packages (Ubuntu/Debian)
```bash
sudo apt install ffmpeg libgl1
```

### Python packages
```bash
pip install ultralytics faiss-cpu flask flask-socketio \
            torch torchvision opencv-python requests \
            insightface onnxruntime lap
```

> **First run downloads:**
> - `yolov8n.pt` — ~6 MB (auto-downloaded by ultralytics)
> - `buffalo_l.zip` — ~300 MB (auto-downloaded by insightface)
> - `resnet50` weights — ~100 MB (auto-downloaded by torchvision)

---

## Project Layout

```
reid_poc/
├── main.py                  ← entire system, single file
├── README.md                ← this file
├── alerts.log               ← created on first run
├── trajectory_final.json    ← created on Ctrl+C
└── recordings/
    ├── G-0000/
    │   ├── cam_a_20240401_103045.mp4
    │   └── cam_b_20240401_103112.mp4
    └── G-0001/
        └── cam_a_20240401_103201.mp4
```

---

## Quick Start

### 1. Clone / download

```bash
mkdir reid_poc && cd reid_poc
# place main.py here
```

### 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics faiss-cpu flask flask-socketio \
            torch torchvision opencv-python requests \
            insightface onnxruntime lap
```

### 3. Run

```bash
python3 main.py \
  --cam_a "rtsp://user:pass@192.168.1.101:554/stream1" \
  --cam_b "rtsp://user:pass@192.168.1.102:554/stream1" \
  --port  5000
```

### 4. Open dashboard

```
http://localhost:5000
```

Or from any device on the same LAN:

```
http://<your-machine-ip>:5000
```

---

## CLI Reference

```
python3 main.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--cam_a` | *(required)* | RTSP URL for Camera A |
| `--cam_b` | *(required)* | RTSP URL for Camera B |
| `--model` | `yolov8n.pt` | YOLOv8 weights — `n/s/m/l.pt` (larger = more accurate, slower) |
| `--conf` | `0.40` | YOLO detection confidence threshold |
| `--sim` | `0.75` | Body ReID cosine similarity threshold |
| `--face_sim` | `0.45` | Face cosine similarity threshold |
| `--webhook` | `None` | Slack / Teams / custom HTTP webhook URL |
| `--sound` | `None` | Path to `.wav` file for audio alert |
| `--port` | `5000` | Flask dashboard port |

---

## Common RTSP URL Formats

| Brand | URL pattern |
|---|---|
| **Hikvision** | `rtsp://user:pass@IP:554/Streaming/Channels/101` |
| **Dahua** | `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0` |
| **Reolink** | `rtsp://user:pass@IP:554/h264Preview_01_main` |
| **TP-Link Tapo** | `rtsp://user:pass@IP:554/stream1` |
| **Generic ONVIF** | `rtsp://user:pass@IP:554/onvif/media` |

> **Special characters in passwords** — URL-encode them:
> `@` → `%40`, `#` → `%23`, `!` → `%21`, `$` → `%24`
>
> Example: password `Mantra@123` becomes `Mantra%40123`

---

## Tuning Guide

### Detection is missing people
```bash
--conf 0.30    # lower threshold catches more people (more false positives)
--model yolov8s.pt  # larger model = better detection
```

### Same person gets multiple body IDs (ID splits)
```bash
--sim 0.65     # lower threshold = easier to match (more false merges)
```

### Different people share one body ID (false merges)
```bash
--sim 0.85     # higher threshold = stricter matching
```

### Face ID keeps resetting
```bash
--face_sim 0.35   # lower threshold = easier face match across frames
```

### Slow inference on CPU

Edit `INFERENCE_SKIP` in `main.py`:
```python
INFERENCE_SKIP = 3   # run ReID every 3rd frame instead of every 2nd
```

Or use a smaller YOLO model:
```bash
--model yolov8n.pt   # nano — fastest
```

### Camera topology — add more cameras

Edit `CAMERA_TOPOLOGY` in `main.py`:
```python
# 3-camera linear layout: entrance → aisle → exit
CAMERA_TOPOLOGY = {
    "cam_a": {"cam_b"},
    "cam_b": {"cam_a", "cam_c"},
    "cam_c": {"cam_b"},
}
```
Then add `"cam_c"` to the `cameras` dict in `main()`.

---

## Output Files

### `alerts.log`
```
2024-04-01 10:30:45  [NEW PERSON] G-0000  face=F-0000 on CAM_A @ 10:30:45
2024-04-01 10:31:12  [HANDOFF] G-0000  face=F-0000  CAM_A -> CAM_B @ 10:31:12
```

### `trajectory_final.json`
```json
[
  {
    "global_id": "G-0000",
    "face_id": "F-0000",
    "trajectory": ["cam_a", "cam_b"],
    "cameras_seen": ["cam_a", "cam_b"],
    "last_camera": "cam_b"
  }
]
```

### `recordings/G-XXXX/`
Per-person MP4 clips at native camera resolution.
- **Lead-in:** 3 seconds of frames before first detection
- **Tail:** 5 seconds after last detection
- **Minimum clip length:** ~8 seconds
- **Format:** H.264 MP4, native camera resolution

---

## Dashboard

| Element | Description |
|---|---|
| **Body IDs** | Total unique persons tracked across all cameras |
| **Face IDs** | Total unique faces identified |
| **Handoffs** | Total cross-camera transitions detected |
| **Live feeds** | MJPEG stream from each camera, annotated in real time |
| **Event log** | Rolling list of NEW PERSON and HANDOFF events with timestamps |

### On-screen labels
```
┌─────────────────────┐
│ G-0003  F-0001      │  ← body ID (blue box) + face ID
│                     │
│     [person body]   │
│   ┌──────────┐      │
│   │  F-0001  │      │  ← teal box drawn around detected face
│   └──────────┘      │
└─────────────────────┘
```

- **G-XXXX** = global body ID (persistent across cameras)
- **F-XXXX** = face identity (ArcFace embedding, camera-independent)

---

## Webhook Payload

Sent as JSON POST to `--webhook` URL on every handoff or new person event:

```json
{
  "text":          "[HANDOFF] G-0003  face=F-0001  CAM_A -> CAM_B @ 10:31:12",
  "type":          "handoff",
  "global_id":     3,
  "face_id":       1,
  "from_cam":      "cam_a",
  "to_cam":        "cam_b",
  "timestamp":     1712345678.0,
  "total_persons": 5,
  "total_faces":   4
}
```

Compatible with Slack incoming webhooks, Microsoft Teams connectors, and any HTTP endpoint.

---

## Troubleshooting

### Camera fails to connect / 30-second timeout
```bash
# Test URL directly with ffplay
ffplay "rtsp://user:pass@IP:554/stream"

# Force TCP (already set in code, but test manually)
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
python3 main.py --cam_a "rtsp://..."
```

### `'Conv' object has no attribute 'bn'`
This happens when `lap` installs mid-run. Fix:
```bash
pip install lap
python3 main.py ...   # restart
```

### `ModuleNotFoundError: No module named 'capture'`
You have an old version of `main.py`. This version is fully self-contained — no other files needed.

### InsightFace spam on startup (`Applied providers / find model`)
This is printed at the C level by ONNX Runtime and is suppressed via fd-level redirect in the current version. If you still see it, your `main.py` may be outdated.

### `error while decoding MB ... bytestream -5`
Harmless H.264 packet loss glitch. Suppressed in the current version via FFmpeg `err_detect;ignore_err` flag. Video quality is unaffected.

### Dashboard shows blank camera feed
The MJPEG stream at `/feed/cam_a` and `/feed/cam_b` only starts once the camera connects. Wait a few seconds after startup.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4-core (Intel i5 / Ryzen 5) | 8-core |
| RAM | 4 GB | 8 GB |
| GPU | Not required | CUDA GPU (3-5× faster) |
| Network | 100 Mbps LAN | Gigabit LAN |
| Python | 3.10 | 3.11+ |
| OS | Ubuntu 20.04+ / Windows 10+ / macOS 12+ | Ubuntu 22.04 |

> **GPU acceleration:** If a CUDA GPU is available, both ResNet-50 (body embedding) and InsightFace (face embedding) will automatically use it. No code changes needed — detected at startup.

---

## License

MIT — free to use, modify, and deploy.
