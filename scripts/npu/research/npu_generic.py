"""Run one RKNN model on the device through caldec_npu_server (generic: inputs by name, float32 / int64) and time it.
Usage: npu_generic.py <server_dir_or_binary> <model.rknn> <inputs.npz> <outputs.npz> [repeats]
(server: caldec_npu_server for fp16 zero-copy, or npu_run_generic for any model incl. int8)
inputs.npz: arrays named like the model inputs, with a leading case axis (case i = arr[i])."""
import os, struct, subprocess, sys, time, tempfile
import numpy as np
srv_dir, model, inp, outp = sys.argv[1:5]; rep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
d = tempfile.mkdtemp(); mt = os.path.join(d, "models.txt")
open(mt, "w").write("1\n1 %s - 0\n" % os.path.abspath(model))
exe = srv_dir if os.path.isfile(srv_dir) else os.path.join(srv_dir, "caldec_npu_server")
p = subprocess.Popen([exe, mt], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
names = []
while True:
    l = p.stdout.readline().decode()
    if not l: sys.exit("server exited")
    if l.strip() == "READY": break
    f = l.split(); names = [x.split(":") for x in f[3:] if ":" in x]
print("inputs", names, flush=True)
X = dict(np.load(inp)); n = len(next(iter(X.values()))); outs, ms = [], []
for i in range(n):
    for r in range(rep):
        w = p.stdin; w.write(struct.pack("<i", 1))
        for nm, typ, el in names:
            a = np.ascontiguousarray(X[nm][i], np.int64 if typ == "i64" else np.float32); assert a.size == int(el), (nm, a.shape, el)
            w.write(a.tobytes())
        w.flush(); st, no = struct.unpack("<ii", p.stdout.read(8)); ys = []
        for _ in range(no):
            (k,) = struct.unpack("<i", p.stdout.read(4)); ys.append(np.frombuffer(p.stdout.read(4 * k), np.float32).copy())
        (t,) = struct.unpack("<f", p.stdout.read(4)); assert st == 0, st; ms.append(t)
    outs.append(ys)
p.stdin.write(struct.pack("<i", 0)); p.stdin.flush(); p.wait(10)
np.savez(outp, **{"out%d" % j: np.stack([o[j] for o in outs]) for j in range(len(outs[0]))})
ms = np.array(ms); print(f"cases {n} x {rep}: ms mean {ms.mean():.2f} median {np.median(ms):.2f} min {ms.min():.2f}")
