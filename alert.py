"""
Alert engine — listens for 'handoff' events on the event bus
and fires desktop notification + webhook + sound + log entry.
"""

import threading
import queue
import time
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger("alert")

# ── Optional integrations ────────────────────────────────────────────────────
try:
    from playsound import playsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Per-person cooldown so the same handoff doesn't fire 30 times ────────────
ALERT_COOLDOWN_SEC = 10   # minimum seconds between alerts for the same global ID


class AlertEngine(threading.Thread):
    """
    Consumes events from event_bus (Queue).
    Expected event dict shape:
        {
          "type": "handoff",         # or "new_person"
          "global_id": 42,
          "from_cam": "cam_a",
          "to_cam":   "cam_b",
          "timestamp": 1712345678.0,
        }
    """
    def __init__(self, event_bus: queue.Queue,
                 webhook_url: str = None,
                 alert_sound: str = None,
                 log_path: str = "alerts.log"):
        super().__init__(daemon=True)
        self.bus          = event_bus
        self.webhook_url  = webhook_url
        self.alert_sound  = alert_sound       # path to a .wav/.mp3 file
        self.log_path     = Path(log_path)
        self._last_alerted = {}               # gid → last alert timestamp
        self._stop        = threading.Event()

        logging.basicConfig(
            filename=str(self.log_path),
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
        )

    def run(self):
        while not self._stop.is_set():
            try:
                event = self.bus.get(timeout=0.5)
            except queue.Empty:
                continue

            if event.get("type") not in ("handoff", "new_person"):
                self.bus.task_done()
                continue

            gid = event["global_id"]
            now = event.get("timestamp", time.time())

            # Cooldown check
            if now - self._last_alerted.get(gid, 0) < ALERT_COOLDOWN_SEC:
                self.bus.task_done()
                continue
            self._last_alerted[gid] = now

            self._fire(event)
            self.bus.task_done()

    def _fire(self, event):
        gid     = event["global_id"]
        ev_type = event["type"]
        ts_str  = datetime.fromtimestamp(event.get("timestamp", time.time()))\
                          .strftime("%H:%M:%S")

        if ev_type == "handoff":
            msg = (f"[HANDOFF] G-{gid:04d}  "
                   f"{event['from_cam'].upper()} → {event['to_cam'].upper()}  "
                   f"at {ts_str}")
        else:
            msg = f"[NEW PERSON] G-{gid:04d} detected at {event.get('cam_id','?')} at {ts_str}"

        # 1. Console
        print(f"\033[93m{msg}\033[0m")   # yellow terminal output

        # 2. Log file
        log.info(msg)

        # 3. Desktop notification (cross-platform best-effort)
        self._desktop_notify(f"ReID Alert", msg)

        # 4. Webhook (Slack, Teams, custom HTTP endpoint)
        if self.webhook_url and HAS_REQUESTS:
            self._webhook(msg, event)

        # 5. Sound
        if self.alert_sound and HAS_SOUND:
            threading.Thread(
                target=playsound, args=(self.alert_sound,), daemon=True
            ).start()

    def _desktop_notify(self, title: str, body: str):
        """Best-effort desktop notification — works on Linux/macOS/Windows."""
        try:
            import platform
            sys = platform.system()
            if sys == "Linux":
                import subprocess
                subprocess.Popen(["notify-send", title, body])
            elif sys == "Darwin":
                import subprocess
                subprocess.Popen([
                    "osascript", "-e",
                    f'display notification "{body}" with title "{title}"'
                ])
            elif sys == "Windows":
                from win10toast import ToastNotifier
                ToastNotifier().show_toast(title, body, duration=4, threaded=True)
        except Exception:
            pass   # notifications are best-effort, never crash the pipeline

    def _webhook(self, msg: str, event: dict):
        payload = {
            "text": msg,
            "global_id": event["global_id"],
            "event_type": event["type"],
            "from_cam": event.get("from_cam"),
            "to_cam":   event.get("to_cam"),
        }
        try:
            requests.post(self.webhook_url, json=payload, timeout=3)
        except Exception as e:
            print(f"[alert] Webhook failed: {e}")

    def stop(self):
        self._stop.set()