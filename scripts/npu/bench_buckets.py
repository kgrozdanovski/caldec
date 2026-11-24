"""Run a bucketed CalDec deployment on the board and score it.
Usage: rknn-venv/bin/python bench_buckets.py <model pattern with {S}> <tag> <buckets> [--npz out/cases_<tag>_s256.npz]
e.g.   bench_buckets.py caldec_npu_npue2_s{S}_int8.rknn eval400 128,160,192,256
Copies each bucket model to the board ($BOARD), runs it on its cases (out/cases_raw_npu_<tag>_b<S>), reassembles the logits in the
original order and prints accuracy vs the fp32 references plus the latency weighted over all questions."""
from pathlib import Path
import json, subprocess, sys, numpy as np, re, os, tempfile
pat, tag, bks = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
npz = sys.argv[sys.argv.index("--npz") + 1] if "--npz" in sys.argv else f"out/cases_{tag}_s256.npz"
B = str(Path(__file__).resolve().parents[0]); os.chdir(B)
K = os.environ.get("SSH_OPTS", "").split()      # extra ssh options (key, known_hosts)
H = os.environ["BOARD"]                          # user@host of the board
assign = json.load(open(f"out/buckets_{tag}.json")); c = np.load(npz); N = c["input_ids"].shape[0]
logits = np.zeros((N, 6), np.float32); lat = np.zeros(N); rows = []
for S in bks:
    idx = assign[str(S)]
    if not idx: continue
    model = pat.format(S=S); cdir = f"cases_raw_npu_{tag}_b{S}"
    subprocess.run(["bash", "-c", f"tar cf - -C out {model} {cdir} | ssh {' '.join(K)} {H} 'cd npu-bench && tar xf -'"], check=True)
    n = len(idx); runs = max(n, 8)
    r = subprocess.run(["ssh", *K, H, f"cd npu-bench && ./rknn_caldec_bench {model} {cdir} {n} {runs} bk.bin 2>&1 | grep -E 'cpu_ms|p50|rror'; rm -f {model}"],
                       capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as td:                  # the logits, copied back from the board
        subprocess.run(["scp", "-q", *K, f"{H}:npu-bench/bk.bin", f"{td}/bk.bin"], check=True)
        y = np.fromfile(f"{td}/bk.bin", np.float32).reshape(-1, 6)[:n]
    p50 = float(re.search(r"p50_ms ([\d.]+)", r).group(1)); p95 = float(re.search(r"p95_ms ([\d.]+)", r).group(1))
    cpu = re.search(r"cpu_ms_per_run_in_rknn_run ([\d.]+)", r)
    for j, i in enumerate(idx): logits[i] = y[j]; lat[i] = p50
    rows.append((S, n, p50, p95, float(cpu.group(1)) if cpu else -1)); print(f"bucket {S}: {n} cases, p50 {p50} ms, p95 {p95} ms, cpu {rows[-1][4]} ms", flush=True)
def sm(z): z = z - z.max(); e = np.exp(z); return e / e.sum()
agree, dps, flips = 0, [], []
for i in range(N):
    k = int(c["marker_mask"][i].sum()); p = sm(logits[i, :k].astype(np.float64)); q = sm(c["ref"][i][0][:k].astype(np.float64))
    a = p.argmax() == q.argmax(); agree += a; dps.append(float(np.abs(p - q).max()))
    if not a: qs = np.sort(q)[::-1]; flips.append({"case": i, "ref_margin": round(float(qs[0] - qs[1]), 3)})
print(json.dumps({"cases": N, "argmax_agreement": f"{agree}/{N}", "max_abs_dp": round(max(dps), 4), "median_abs_dp": round(float(np.median(dps)), 4),
                  "p99_abs_dp": round(float(np.percentile(dps, 99)), 4), "flips": flips,
                  "latency_mean_ms": round(float(lat.mean()), 1), "latency_median_ms": round(float(np.median(lat)), 1),
                  "latency_p95_ms": round(float(np.percentile(lat, 95)), 1)}))
np.save(f"out/logits_{os.path.basename(pat).replace('{S}','B')}_{tag}.npy", logits)
