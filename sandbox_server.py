"""Local server for the 3D sandbox. Run: .venv/Scripts/python sandbox_server.py  -> http://127.0.0.1:8765
Serves sandbox/ and runs full brain+physics simulations (run.py --record) one at a time."""
import json, pathlib, re, subprocess, sys, threading, uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).parent.resolve()
STATIC = ROOT / "sandbox"
RUNS = ROOT / "results" / "sandbox"
LIMITS = {"reward": (-1, 1), "punish": (-1, 1), "octopamine": (-1, 1), "danger": (0, 4),
          "seconds": (1, 10), "seed": (0, 10**6)}
jobs, lock = {}, threading.Lock()


def clamp(v, lo, hi):
    return min(max(float(v), lo), hi)


def start_job(p):
    with lock:
        if any(j["proc"].poll() is None for j in jobs.values()):
            return None
        jid = uuid.uuid4().hex[:10]
        out = RUNS / jid
        out.mkdir(parents=True)
        cmd = [sys.executable, str(ROOT / "run.py"), "--record", str(out),
               "--dopamine-reward", str(p["reward"]), "--dopamine-punish", str(p["punish"]),
               "--octopamine", str(p["octopamine"]), "--danger", str(p["danger"]),
               "--seconds", str(p["seconds"]), "--seed", str(int(p["seed"])),
               "--food", str(p["food"][0]), str(p["food"][1]),
               "--danger-pos", str(p["danger_pos"][0]), str(p["danger_pos"][1])]
        if p.get("heading") is not None:
            cmd += ["--heading", str(p["heading"])]
        log = open(out / "log.txt", "w")
        jobs[jid] = {"proc": subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), "dir": out,
                     "seconds": p["seconds"]}
        return jid


def status(jid):
    j = jobs.get(jid)
    if not j:
        return {"status": "unknown"}
    text = (j["dir"] / "log.txt").read_text(errors="replace")
    ts = re.findall(r"^t=([\d.]+)s", text, re.M)
    code = j["proc"].poll()
    st = "running" if code is None else ("done" if code == 0 and (j["dir"] / "run.json").exists() else "error")
    return {"status": st, "progress": (float(ts[-1]) / j["seconds"]) if ts else 0.0,
            "error": text[-1500:] if st == "error" else None}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(STATIC), **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        m = re.fullmatch(r"/api/run/([0-9a-f]{10})", self.path)
        if m:
            return self.send_json(status(m.group(1)))
        m = re.fullmatch(r"/runs/([0-9a-f]{10})/(run\.json|poses\.bin|spikes_idx\.bin|spikes_cnt\.bin)", self.path)
        if m:
            f = RUNS / m.group(1) / m.group(2)
            if not f.exists():
                return self.send_error(404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if f.suffix == ".json" else "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/run":
            return self.send_error(404)
        try:
            raw = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            p = {k: clamp(raw.get(k, 0 if k != "seconds" else 6), *lim) for k, lim in LIMITS.items()}
            p["food"] = [clamp(v, -60, 60) for v in raw["food"]][:2]
            p["danger_pos"] = [clamp(v, -60, 60) for v in raw["danger_pos"]][:2]
            p["heading"] = clamp(raw["heading"], -7, 7) if raw.get("heading") is not None else None
        except (KeyError, TypeError, ValueError) as e:
            return self.send_json({"error": f"bad request: {e}"}, 400)
        jid = start_job(p)
        if jid is None:
            return self.send_json({"error": "a simulation is already running"}, 409)
        self.send_json({"id": jid})


if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("sandbox on http://127.0.0.1:8765", flush=True)
    srv.serve_forever()
