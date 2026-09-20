"""Serve a local Snake arena: python -m examples.gaming.snake_arena --help."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import platform
import secrets
import statistics
import threading
import time

import numpy as np

from .game import DIRECTIONS, SnakeGame
from .policy import JevPlayer, LayaPlayer, System1Player, planner_input

HERE = Path(__file__).resolve().parent


def settings(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an object")
    mode = payload.get("mode", "timed")
    if mode not in ("timed", "turns"):
        raise ValueError("Choose timed or turns")
    values = {"mode": mode}
    for key, default, low, high in (("seed", 2026, 0, 2**31 - 1), ("seconds", 30, 5, 60), ("turns", 120, 20, 600)):
        value = payload.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
        values[key] = value
    shield = payload.get("shield", True)
    if type(shield) is not bool:
        raise ValueError("shield must be Boolean")
    return values | {"shield": shield, "width": 24, "height": 16, "initial_length": 6}


def apply_choice(result, plan, shield):
    probabilities = result.get("probabilities", {})
    if set(probabilities) != set(DIRECTIONS) or any(
        type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
    ) or abs(sum(probabilities.values()) - 1) > .02:
        raise ValueError("Provider returned invalid direction probabilities; no move executed")
    proposed = max(DIRECTIONS, key=probabilities.__getitem__)
    if shield and not plan["safe"]:
        raise ValueError("Planner invariant failed: no safe direction")
    executed = max(plan["safe"], key=probabilities.__getitem__) if shield else proposed
    return proposed, executed


def load_key(path):
    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
    if key or path is None:
        return key
    # Data only; never source or execute a credential file.
    for line in path.read_text().splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
            return value.strip().strip("\"'")
    return None


class Arena:
    def __init__(self, players, output):
        self.players, self.output = players, output
        self.output.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.running = False
        self.phase = "ready"
        self.run_id = None
        self.config = settings({})
        self.started = self.ended = None
        self.lanes = {}
        self.latest = None
        self.source_sha256 = sha256(b"".join(p.read_bytes() for p in sorted(HERE.glob("*.py")))).hexdigest()
        self._reset_lanes()

    def _reset_lanes(self):
        self.lanes = {}
        for name in self.players:
            board = SnakeGame(seed=self.config["seed"])
            self.lanes[name] = {"board": board.snapshot(), "status": "ready", "last": None,
                                "times": [], "total_times": [], "trace": [], "requests": 0,
                                "interventions": 0, "reviews": 0, "planner_matches": 0,
                                "unsafe_proposals": 0, "teacher_calls": 0, "late_responses": 0,
                                "initial": board.snapshot(), "elapsed": 0, "error": None}

    def start(self, payload):
        config = settings(payload)
        with self.lock:
            if self.running:
                raise ValueError("A race is already running or stopping")
            self.config = config
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
            self.stop_event.clear()
            self._reset_lanes()
            self.running, self.phase = True, "running"
            self.started, self.ended = time.perf_counter() + .1, None
            workers = [threading.Thread(target=self._play, args=(name,), daemon=True) for name in self.players]
            for worker in workers:
                worker.start()
            threading.Thread(target=self._finish, args=(workers,), daemon=True).start()
        return self.snapshot()

    def stop(self):
        with self.lock:
            if self.running:
                self.phase = "stopping"
                self.stop_event.set()

    def _play(self, name):
        player, lane = self.players[name], self.lanes[name]
        game = SnakeGame(seed=self.config["seed"])
        self.stop_event.wait(max(0, self.started - time.perf_counter()))
        deadline = self.started + (self.config["seconds"] if self.config["mode"] == "timed" else 300)
        with self.lock:
            lane["status"] = "live"
        try:
            while not self.stop_event.is_set() and game.alive and not game.won:
                if time.perf_counter() >= deadline or (self.config["mode"] == "turns" and game.ticks >= self.config["turns"]):
                    break
                began = time.perf_counter()
                state, questions, plan = planner_input(game)
                inference_start = time.perf_counter()
                result = player.predict(state, questions)
                inference_ms = (time.perf_counter() - inference_start) * 1000
                with self.lock:
                    lane["requests"] += 1
                    lane["teacher_calls"] += result["teacher_calls"]
                # A reply after the common deadline must not buy this lane an extra move.
                if time.perf_counter() >= deadline or self.stop_event.is_set():
                    with self.lock:
                        lane["late_responses"] += 1
                    break
                proposed, executed = apply_choice(result, plan, self.config["shield"])
                game.step(executed)
                total_ms = (time.perf_counter() - began) * 1000
                event = {"at": time.perf_counter() - self.started, "proposed": proposed, "executed": executed,
                         "probabilities": result["probabilities"], "needs_review": result["needs_review"],
                         "intervened": proposed != executed, "safe": plan["safe"], "preferred": plan["preferred"],
                         "inference_ms": inference_ms, "total_ms": total_ms, "input_tokens": result["input_tokens"],
                         "output_tokens": result["output_tokens"], "head": list(game.head),
                         "food": list(game.food) if game.food else None, "score": game.score,
                         "length": len(game.body), "alive": game.alive, "won": game.won}
                with self.lock:
                    lane["trace"].append(event)
                    lane["times"].append(inference_ms)
                    lane["total_times"].append(total_ms)
                    lane["last"] = event
                    lane["board"] = game.snapshot()
                    lane["interventions"] += proposed != executed
                    lane["reviews"] += result["needs_review"] is True
                    lane["planner_matches"] += proposed == plan["preferred"]
                    lane["unsafe_proposals"] += proposed not in plan["safe"]
        except Exception as error:
            # Only classify errors here; never expose HTTP response bodies or credentials.
            with self.lock:
                lane["error"] = f"{type(error).__name__}: prediction failed; no substituted model or result"
        finally:
            with self.lock:
                lane["elapsed"] = max(0, min(time.perf_counter(), deadline) - self.started)
                lane["status"] = ("error" if lane["error"] else "dead" if not game.alive else "cleared" if game.won
                                  else "stopped" if self.stop_event.is_set() else "finished")

    def _finish(self, workers):
        for worker in workers:
            worker.join()
        with self.lock:
            self.ended = time.perf_counter()
            self.phase = "stopped" if self.stop_event.is_set() else "finished"
            self.running = False
            self.latest = self.report()
            self.latest["summary"]["has_replay"] = True
            raw = json.dumps(self.latest, indent=2) + "\n"
            (self.output / f"{self.run_id}.json").write_text(raw)
            (self.output / "latest.json").write_text(raw)

    def snapshot(self):
        with self.lock:
            elapsed = max(0, (self.ended or time.perf_counter()) - self.started) if self.started else 0
            if self.config["mode"] == "timed":
                elapsed = min(elapsed, self.config["seconds"])
            lanes = {}
            for name, lane in self.lanes.items():
                times = lane["times"]
                spent = lane["elapsed"] if lane["status"] not in ("live", "ready") else elapsed
                lanes[name] = {key: value for key, value in lane.items() if key not in ("times", "total_times", "trace", "initial")}
                lanes[name].update({"metadata": self.players[name].metadata,
                                    "p50_ms": statistics.median(times) if times else None,
                                    "p95_ms": float(np.quantile(times, .95)) if times else None,
                                    "rate": lane["board"]["ticks"] / spent if spent > 0 else 0,
                                    "total_p50_ms": statistics.median(lane["total_times"]) if times else None})
            return {"run_id": self.run_id, "phase": self.phase, "running": self.running,
                    "settings": self.config, "elapsed": elapsed, "lanes": lanes,
                    "has_replay": self.latest is not None}

    def report(self):
        return {"format": "system1.snake-arena.v1", "summary": self.snapshot(),
                "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
                "source_sha256": self.source_sha256,
                "game_source": "mizchi/laya-mlx@dc3aa6b150cb861d0788fbd421cfd1303de4ed57",
                "protocol": {
                    "inputs": "Same compact planner-generated state and three questions for every provider; no raw-board reasoning claim",
                    "system1": "Taught on synthetic planner descriptions; field-namespaced words; no saved decision cache",
                    "execution": "Raw argmax evaluated even when System1 requests review; optional identical cycle shield for every player",
                    "clock": "Common wall-clock start; timed mode rejects responses arriving after deadline; turns mode gives equal move budgets",
                    "timing": "Model loading and warm-up excluded; inference includes encoding and result formatting; end-to-end includes planning and game update; loops run concurrently",
                    "food": "Same seed and initial board; subsequent food placement depends on each lane's occupied cells",
                    "baseline": "The shared planner already computes a safe preferred move; a model is not necessary to solve this game",
                },
                "lanes": {name: {"initial": lane["initial"], "events": lane["trace"]} for name, lane in self.lanes.items()}}


class RecordedArena:
    """Show an existing report without loading models, credentials or network clients."""
    def __init__(self, path):
        raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        self.latest = json.loads(raw)
        if self.latest.get("format") != "system1.snake-arena.v1":
            raise ValueError("Expected a Snake arena v1 report")
        self.token = secrets.token_urlsafe(32)

    def snapshot(self):
        return self.latest["summary"] | {"running": False, "phase": "recorded result", "read_only": True, "has_replay": True}

    def start(self, payload):
        raise ValueError("Replay-only server: start the live arena to make new decisions")

    def stop(self):
        pass


def serve(arena, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Localhost only"})
            if self.path == "/":
                page = (HERE / "index.html").read_text().replace("__ARENA_TOKEN__", arena.token)
                return self.send(200, page.encode(), "text/html; charset=utf-8")
            if self.path == "/api/status":
                return self.send(200, arena.snapshot())
            if self.path in ("/api/export", "/api/replay"):
                if arena.latest is None:
                    return self.send(404, {"error": "Complete a run first"})
                return self.send(200, arena.latest)
            if self.path == "/favicon.ico":
                return self.send(204, b"", "image/x-icon")
            return self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_host() or not secrets.compare_digest(self.headers.get("X-Arena-Token", ""), arena.token):
                return self.send(403, {"error": "Use the local arena page"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("Request too large")
                payload = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/api/start":
                    return self.send(200, arena.start(payload))
                if self.path == "/api/stop":
                    arena.stop()
                    return self.send(200, arena.snapshot())
                return self.send(404, {"error": "Not found"})
            except (ValueError, TypeError) as error:
                return self.send(400, {"error": str(error)})
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Snake arena ready: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        arena.stop()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Local Laya-MLX English checkpoint directory")
    parser.add_argument("--replay", type=Path, help="Show a recorded JSON or JSON.gz report without models or API access")
    parser.add_argument("--credentials", type=Path, help="Ignored local file containing TYPESAFE_API_KEY; never sent to browser")
    parser.add_argument("--output", type=Path, default=Path(".system1/snake"))
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    if args.replay:
        if args.model or args.credentials:
            parser.error("--replay does not use --model or --credentials")
        serve(RecordedArena(args.replay), args.port)
        return
    if args.model is None:
        parser.error("live mode requires --model")
    print("Teaching System1's compact planner-reading skill...", flush=True)
    players = {"system1": System1Player(args.output / "snake.s1m")}
    print("Loading Laya through MLX...", flush=True)
    players["laya"] = LayaPlayer(args.model)
    players["jev"] = JevPlayer(load_key(args.credentials))
    # Warm-up is explicit and outside every run. Jev receives one real warm-up request.
    for name, player in players.items():
        print(f"Warming {name}...", flush=True)
        for seed in ([11, 12, 13, 14] if name == "laya" else [11]):
            game = SnakeGame(seed=seed)
            state, questions, _ = planner_input(game)
            player.predict(state, questions)
    serve(Arena(players, args.output), args.port)


if __name__ == "__main__":
    main()
