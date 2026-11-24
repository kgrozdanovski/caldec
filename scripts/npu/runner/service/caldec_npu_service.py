"""CalDec NPU service: JSON lines over a Unix socket, one CalDecNPU (one NPU model set) shared by all clients.

Started by systemd socket activation (caldec-npu.socket passes the listening socket as fd 3). Each connection sends
one JSON object per line, {"state": ..., "questions": {...}}, and gets one JSON answer per line, in order;
{"error": "..."} for a bad request. Requests are served one at a time (one NPU). The process exits after IDLE
seconds without a connection; the next connection starts it again (about 6 s to load the 10 models).

Usage outside systemd (tests): caldec_npu_service.py <model_dir> <socket_path>
Env: CALDEC_NPU_DIR (default /opt/caldec-npu), CALDEC_NPU_IDLE (seconds, default 600, 0 = never exit).
"""
import json, os, socket, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caldec_npu import CalDecNPU   # noqa: E402

MAX_LINE, MAX_CONN, READ_TIMEOUT = 1 << 20, 8, 120
DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CALDEC_NPU_DIR", "/opt/caldec-npu")
IDLE = float(os.environ.get("CALDEC_NPU_IDLE", "600"))


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def listener():
    if os.environ.get("LISTEN_FDS") == "1" and os.environ.get("LISTEN_PID") == str(os.getpid()):
        return socket.socket(fileno=3)
    path = sys.argv[2]
    if os.path.exists(path):
        os.unlink(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.bind(path); os.chmod(path, 0o600); s.listen(MAX_CONN)
    return s


class Service:
    def __init__(self):
        t0 = time.perf_counter(); self.m = CalDecNPU(DIR)
        log("models loaded in %.1f s (buckets %s, act head %s)" % (time.perf_counter() - t0, self.m.sizes,
                                                                   "yes" if self.m.act_head else "no"))
        self.npu = threading.Lock(); self.state = threading.Lock(); self.active = 0; self.last = time.monotonic()
        self.served = 0

    def answer(self, line):
        try:
            req = json.loads(line)
            if not isinstance(req, dict) or not isinstance(req.get("questions"), dict) or "state" not in req:
                raise ValueError('expected {"state": ..., "questions": {...}}')
            t0 = time.perf_counter()
            with self.npu:
                out = self.m.system_one(req["state"], req["questions"])
            out["usage"]["total_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            self.served += 1
            return out
        except Exception as e:   # a bad request must not take the service down
            return {"error": "%s: %s" % (type(e).__name__, e)}

    def client(self, conn):
        with self.state:
            self.active += 1
        try:
            conn.settimeout(READ_TIMEOUT); f = conn.makefile("rb"); w = conn.makefile("wb")
            while True:
                line = f.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE:
                    w.write(b'{"error": "request line longer than 1 MiB"}\n'); w.flush(); break
                if line.strip():
                    w.write((json.dumps(self.answer(line)) + "\n").encode()); w.flush()
        except (OSError, socket.timeout) as e:
            log("client: %s" % e)
        finally:
            conn.close()
            with self.state:
                self.active -= 1; self.last = time.monotonic()

    def run(self, srv):
        srv.settimeout(10)
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                with self.state:
                    idle = self.active == 0 and IDLE > 0 and time.monotonic() - self.last > IDLE
                if idle:
                    log("idle for %.0f s after %d requests: exiting (socket activation restarts it)" % (IDLE, self.served))
                    return
                continue
            with self.state:
                busy = self.active >= MAX_CONN
            if busy:
                conn.sendall(b'{"error": "too many connections"}\n'); conn.close(); continue
            threading.Thread(target=self.client, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    srv = listener(); svc = Service()
    try:
        svc.run(srv)
    finally:
        svc.m.close()
