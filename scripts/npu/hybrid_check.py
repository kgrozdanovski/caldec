from pathlib import Path
import os, sys, numpy as np, torch
os.environ["CALDEC_DENSE_ATTENTION"]="1"; sys.argv=[sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parents[0])); import export_caldec as E
from data import load_agent
torch.backends.mha.set_fastpath_enabled(False)
m = load_agent(E.CKPT, device="cpu").model.eval()
S,D=256,768; sizes=[6]+[S*D]*6; per=sum(sizes); raw=np.fromfile("out/taps.bin",np.float32)
dt={"input_ids":np.int64,"marker_pos":np.int64,"qtype":np.int64}
def ld(j,k,shape): return torch.from_numpy(np.fromfile(f"out/cases_raw_m_s256/{j}_{k}.bin",dt.get(k,np.float32)).reshape(shape))
def sm(z): z=z-z.max(); e=np.exp(z); return e/e.sum()
with torch.inference_mode():
    for j in range(3):
        h = torch.from_numpy(raw[j*per+sum(sizes[:6]) : j*per+per].reshape(1,S,D).copy())   # device final_norm tap
        ref = np.load(f"out/taps_ref_{j}.npz")["arr_0"][0]
        q, hm, mp, mm = ld(j,"qtype",(1,)), ld(j,"head_mask",(S,S)), ld(j,"marker_pos",(1,6)), ld(j,"marker_mask",(1,6))
        x = h + m.type_emb(q)[:, None, :]
        for layer in m.head.layers: x = layer(x, src_mask=hm)
        lg = (m.scorer(torch.gather(x,1,mp[:,:,None].expand(-1,-1,D))).squeeze(-1) + (mm-1)*1e4).numpy()[0]
        k=int(mm.sum()); p,r=sm(lg[:k]),sm(ref[:k])
        print(f"case {j}: NPU-encoder+CPU-head p={np.round(p,3)}  ref p={np.round(r,3)}  argmax {'MATCH' if p.argmax()==r.argmax() else 'DIFF'}")
