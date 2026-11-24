"""Interactive client for npu_run_generic / caldec_npu_server (same protocol): NPU(exe, model).run({name: array}) -> outputs."""
import os, struct, subprocess, tempfile
import numpy as np
class NPU:
    def __init__(self, exe, model):
        d = tempfile.mkdtemp(); mt = os.path.join(d, "models.txt"); open(mt, "w").write("1\n1 %s - 0\n" % os.path.abspath(model))
        self.p = subprocess.Popen([exe, mt], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while True:
            l = self.p.stdout.readline().decode()
            if not l: raise RuntimeError("server exited")
            if l.strip() == "READY": break
            f = l.split(); self.names = [x.split(":") for x in f[3:] if ":" in x]
        self.ms = []
    def run(self, x):
        w = self.p.stdin; w.write(struct.pack("<i", 1))
        for nm, typ, el in self.names:
            a = np.ascontiguousarray(x[nm], np.int64 if typ == "i64" else np.float32); assert a.size == int(el), (nm, a.shape, el); w.write(a.tobytes())
        w.flush(); r = self.p.stdout; st, no = struct.unpack("<ii", r.read(8)); ys = []
        for _ in range(no):
            (k,) = struct.unpack("<i", r.read(4)); ys.append(np.frombuffer(r.read(4 * k), np.float32).copy())
        (t,) = struct.unpack("<f", r.read(4)); self.ms.append(t)
        if st: raise RuntimeError("rknn error %d" % st)
        return ys
    def close(self):
        self.p.stdin.write(struct.pack("<i", 0)); self.p.stdin.flush(); self.p.wait(10)
