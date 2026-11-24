from pathlib import Path
import os, sys, numpy as np, torch
os.environ["CALDEC_DENSE_ATTENTION"]="1"; sys.argv=[sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[0])); import export_caldec as E
from data import load_agent
torch.backends.mha.set_fastpath_enabled(False)
agent = load_agent(E.CKPT, device="cpu"); agent.model.encoder.config._attn_implementation="eager"
cm = E.CoreMasks(agent.model).eval()
c = np.load("out/cases_s256.npz"); S=256
names=["input_ids","local_mask","global_mask","head_mask","marker_pos","marker_mask","qtype"]
sizes={"input_ids":(1,S),"local_mask":(1,1,S,S),"global_mask":(1,1,S,S),"head_mask":(S,S),"marker_pos":(1,6),"marker_mask":(1,6),"qtype":(1,)}
dt={"input_ids":np.int64,"marker_pos":np.int64,"qtype":np.int64}
mx={}
def hook(name):
    def f(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        if torch.is_tensor(t): mx[name]=max(mx.get(name,0), float(t.detach().float().abs().max()))
    return f
for n, m in cm.named_modules():
    if n.count('.') <= 4 and n: m.register_forward_hook(hook(n))
dev = "cpu"
cmh = cm.to(dev).half()
def sm(z): z=z-z.max(); e=np.exp(z); return e/e.sum()
ag=0
with torch.inference_mode():
    for i in range(20):
        f=[torch.from_numpy(np.fromfile(f'out/cases_raw_m_s256/{i}_{k}.bin',dt.get(k,np.float32)).reshape(sizes[k])).to(dev) for k in names]
        f=[x.half() if x.dtype==torch.float32 else x for x in f]
        y=cmh(*f).float().cpu().numpy()[0]; k=int(c['marker_mask'][i].sum()); ag+= sm(y[:k]).argmax()==sm(c['ref'][i][0][:k]).argmax()
print("torch fp16 on", dev, "agree", ag, "/20")
top = sorted(mx.items(), key=lambda kv: -kv[1])[:8]
for k, v in top: print(f"  max|act| {v:10.1f}  {k}")
