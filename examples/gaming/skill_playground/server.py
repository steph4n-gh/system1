"""Local teaching playground: real model calls, no prerecorded decisions."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time

from . import teacher
from .experiment import corrected_rows
from .skills import Network, Single, teach, load
from .world import World


class Playground:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.network = Network(directory)
        self.single = Single(directory)
        self.corrected = False
        self.last_teaching = None
        self.reset({})

    def reset(self, settings):
        allowed = {"seed", "locked", "returning", "purple", "changed"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError("Unknown settings")
        seed = settings.get("seed", 3000)
        if type(seed) is not int or not 0 <= seed <= 1000000:
            raise ValueError("Seed must be an integer from 0 to 1000000")
        config = {k: settings.get(k, True) for k in allowed - {"seed"}}
        if any(type(v) is not bool for v in config.values()):
            raise ValueError("Mission settings must be booleans")
        self.settings = {"seed": seed, **config}
        self.worlds = {name: World(seed, locked=config["locked"], return_home=config["returning"],
                                  purple=config["purple"], purple_dangerous=config["changed"])
                       for name in ("network", "single", "rules")}
        self.decisions = dict.fromkeys(self.worlds)
        self.latencies = {name: [] for name in self.worlds}
        self.reviews = dict.fromkeys(self.worlds, 0)
        return self.snapshot()

    def step(self):
        controllers = {"network": self.network.decide, "single": self.single.decide, "rules": teacher.decide}
        for name, world in self.worlds.items():
            if world.done:
                continue
            started = time.perf_counter()
            decision = controllers[name](world)
            self.latencies[name].append((time.perf_counter()-started)*1000)
            self.reviews[name] += int(decision["review"])
            self.decisions[name] = decision
            if decision["action"] is None:
                world.paused_for_review = True
            else:
                world.step(decision["action"])
        return self.snapshot()

    def correct(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Correction must be a boolean")
        if enabled:
            data = json.loads((self.directory / "lessons.json").read_text())
            rows = corrected_rows(data["terrain"])
            started = time.perf_counter()
            self.network.terrain = teach("terrain", rows, self.directory / "terrain-corrected.s1m")
            self.last_teaching = {"milliseconds": (time.perf_counter()-started)*1000,
                                  "new_examples": 8, "new_calibration": sum(r["split"] == "calibration" and "surface_purple " in r["input"] for r in rows)}
        else:
            self.network.terrain = load(self.directory / "terrain.s1m")
            self.last_teaching = None
        self.corrected = enabled
        return self.reset(self.settings)

    def snapshot(self):
        import statistics
        return {"settings": self.settings, "corrected": self.corrected, "teaching": self.last_teaching,
                "lanes": {name: {"world": world.snapshot(), "decision": self.decisions[name],
                                  "review_steps": self.reviews[name],
                                  "median_ms": statistics.median(self.latencies[name]) if self.latencies[name] else 0}
                          for name, world in self.worlds.items()}}


def serve(directory, port=8789):
    playground = Playground(directory)
    html = Path(__file__).with_name("index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, content, kind="application/json"):
            data = json.dumps(content).encode() if kind == "application/json" else content
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.reply(200, playground.snapshot())
            elif self.path == "/api/report":
                self.reply(200, json.loads((Path(directory) / "report.json").read_text()))
            elif self.path == "/api/lessons":
                self.reply(200, json.loads((Path(directory) / "lessons.json").read_text()))
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            # Browsers cannot mutate the localhost demo from a remote origin.
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}") or (origin and origin not in (f"http://{host}",)):
                return self.reply(403, {"error": "Use the local playground page"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Send a small JSON object")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/reset":
                    state = playground.reset(data)
                elif self.path == "/api/step" and not data:
                    state = playground.step()
                elif self.path == "/api/teach" and set(data) == {"enabled"}:
                    state = playground.correct(data["enabled"])
                else:
                    return self.reply(404, {"error": "Unknown action"})
                self.reply(200, state)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.reply(400, {"error": "Invalid playground request"})

        def log_message(self, *args):
            pass

    print(f"Teaching playground: http://127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
