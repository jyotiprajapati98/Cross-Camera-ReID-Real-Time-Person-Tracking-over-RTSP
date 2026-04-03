"""
Flask + SocketIO web dashboard.
Streams MJPEG frames and pushes live events to connected browsers.
"""

import threading
import queue
import time
import base64
import cv2
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app     = Flask(__name__)
sio     = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared state written by inference threads, read by Flask
_latest_frames  = {}           # cam_id → JPEG bytes
_frame_lock     = threading.Lock()
_stats          = {"persons": 0, "handoffs": 0, "events": []}
_stats_lock     = threading.Lock()


# ── Called from inference thread to push a new frame ─────────────────────────
def push_frame(cam_id: str, frame: np.ndarray):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    jpg = buf.tobytes()
    with _frame_lock:
        _latest_frames[cam_id] = jpg


# ── Called from alert engine to push a live event ────────────────────────────
def push_event(event: dict):
    with _stats_lock:
        if event["type"] == "handoff":
            _stats["handoffs"] += 1
        _stats["persons"] = event.get("total_persons", _stats["persons"])
        entry = {
            "time":      time.strftime("%H:%M:%S",
                         time.localtime(event.get("timestamp", time.time()))),
            "type":      event["type"],
            "global_id": f"G-{event['global_id']:04d}",
            "detail":    (f"{event.get('from_cam','?').upper()} → "
                          f"{event.get('to_cam','?').upper()}")
                         if event["type"] == "handoff"
                         else event.get("cam_id","?").upper(),
        }
        _stats["events"].insert(0, entry)
        _stats["events"] = _stats["events"][:50]   # keep last 50

    sio.emit("event", entry)
    sio.emit("stats", {
        "persons":  _stats["persons"],
        "handoffs": _stats["handoffs"],
    })


# ── MJPEG stream endpoint (one per camera) ────────────────────────────────────
def _mjpeg_gen(cam_id: str):
    while True:
        with _frame_lock:
            jpg = _latest_frames.get(cam_id)
        if jpg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.04)   # ~25 fps cap


@app.route("/feed/<cam_id>")
def video_feed(cam_id):
    return Response(
        _mjpeg_gen(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


def run_dashboard(host="0.0.0.0", port=5000):
    print(f"[dashboard] Starting at http://{host}:{port}")
    sio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)