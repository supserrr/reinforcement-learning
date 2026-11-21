"""3D viewer for the drone delivery environment.

A small local web server streams one JSON message per environment step to a
three.js page (environment/web) over Server-Sent Events. The same server
exposes the integration points a real deployment would use:

    GET  /scene       static map: sites, coordinates, elevations, action names
    GET  /state       the latest step message (JSON)
    GET  /events      live stream of step messages (text/event-stream)
    POST /api/act     {"observation": [38 floats], "action_mask": [10 ints]}
                      -> {"action": int, "name": str} from the loaded policy

Only the Python standard library is used on the server side.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from .custom_env import ACTION_NAMES, SITE_ELEVATION_M, SITE_NAMES, SITE_XY, DroneDeliveryEnv

WEB_DIR = Path(__file__).resolve().parent / "web"


def scene_description(env: DroneDeliveryEnv, agent: str = "", agent_note: str = "") -> dict:
    c = env.cfg
    return {
        "agent": agent,
        "agent_note": agent_note,
        "sites": [
            {"id": i, "name": SITE_NAMES[i], "x": float(SITE_XY[i, 0]), "y": float(SITE_XY[i, 1]),
             "elevation_m": float(SITE_ELEVATION_M[i]), "hub": i == 0}
            for i in range(len(SITE_NAMES))
        ],
        "actions": list(ACTION_NAMES),
        "shift_minutes": c.shift_minutes,
        "cargo_capacity": c.cargo_capacity,
        "emergency_deadline": c.emergency_deadline,
        "routine_deadline": c.routine_deadline,
    }


def step_message(env: DroneDeliveryEnv, *, episode: int, step: int, action: int | None,
                 reward: float, anim_ms: int, agent: str) -> dict:
    """The serialisation format shared by the viewer, the /state endpoint and
    the JSONL episode logs."""
    return {
        "episode": episode,
        "step": step,
        "agent": agent,
        "action": action,
        "action_name": None if action is None else ACTION_NAMES[action],
        "reward": round(float(reward), 3),
        "anim_ms": int(anim_ms),
        "snapshot": env.snapshot(),
    }


class _QuietServer(ThreadingHTTPServer):
    """Browsers drop connections on every reload; keep that out of the agent's terminal log."""

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class Viewer:
    def __init__(self, port: int = 8765, open_browser: bool = True,
                 policy_fn: Callable[[dict], int] | None = None, record_dir: Path | None = None):
        self.record_dir = record_dir
        self.history: list[tuple[str, str]] = []  # replayed to a recording page that connects late
        if record_dir:
            record_dir.mkdir(parents=True, exist_ok=True)
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.scene: dict | None = None
        self.latest: dict | None = None
        self.policy_fn = policy_fn
        handler = partial(_Handler, viewer=self, directory=str(WEB_DIR))
        self.server = _QuietServer(("127.0.0.1", port), handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        if open_browser:
            webbrowser.open(self.url + ("?record" if record_dir else ""))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def set_scene(self, scene: dict):
        self.scene = scene
        self.publish("scene", scene)

    def publish(self, kind: str, payload: dict):
        if kind == "step":
            self.latest = payload
        data = json.dumps(payload)
        if self.record_dir and kind in ("episode_start", "step"):
            self.history.append((kind, data))
        with self.lock:
            for q in list(self.clients):
                q.put((kind, data))

    def wait_for_client(self, timeout: float = 20.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            with self.lock:
                if self.clients:
                    return True
            time.sleep(0.1)
        return False

    def close(self):
        self.server.shutdown()


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, viewer: Viewer, **kwargs):
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def log_message(self, *args):  # keep the terminal for the agent's log
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/scene":
            return self._json(self.viewer.scene or {})
        if path == "/state":
            return self._json(self.viewer.latest or {})
        if path == "/events":
            return self._stream()
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/frame") and self.viewer.record_dir:
            n = int(self.path.split("n=")[-1])
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            (self.viewer.record_dir / f"frame_{n:05d}.jpg").write_bytes(body)
            self.viewer.frames_received = max(getattr(self.viewer, "frames_received", 0), n + 1)
            return self._json({"ok": True})
        if self.path != "/api/act":
            return self._json({"error": "not found"}, 404)
        if self.viewer.policy_fn is None:
            return self._json({"error": "no policy loaded"}, 503)
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            action = int(self.viewer.policy_fn({"observation": req["observation"],
                                                "action_mask": req["action_mask"]}))
            return self._json({"action": action, "name": ACTION_NAMES[action]})
        except Exception as e:  # malformed request
            return self._json({"error": str(e)}, 400)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: queue.Queue = queue.Queue()
        with self.viewer.lock:
            self.viewer.clients.append(q)
        try:
            if self.viewer.scene:
                q.put(("scene", json.dumps(self.viewer.scene)))
            for item in list(self.viewer.history):
                q.put(item)
            while True:
                try:
                    kind, data = q.get(timeout=10)
                    self.wfile.write(f"event: {kind}\ndata: {data}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.viewer.lock:
                if q in self.viewer.clients:
                    self.viewer.clients.remove(q)
