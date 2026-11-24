"""PC: runner.act_probability (numpy) vs packed.forward's act head (torch fp32) on the first 30 packed cases."""
from pathlib import Path
import pickle, sys
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1]) + "/runner")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import export_caldec as ex, packed, caldec_npu as rn
agent = ex.load_agent(ex.CKPT, device="cpu"); m = agent.model.float().eval()
head = {k: v.astype(np.float64) for k, v in np.load(str(Path(__file__).resolve().parents[1]) + "/runner/act_head.npz").items()}
D = pickle.load(open(str(Path(__file__).resolve().parents[1]) + "/packed/cases.pkl", "rb")); worst = 0.0; n = 0
for c in D["cases"][:30]:
    b = packed.collate_cases([c], D["pad"], torch.device("cpu"))
    with torch.inference_mode():
        lg, act = packed.forward(m, b)
        h = m.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        hq = h.reshape(-1, h.size(-1))[b["view"]] + m.type_emb(b["qtype"])[:, None, :]
        for layer in m.head.layers:
            hq = layer(hq, src_key_padding_mask=~b["view_mask"])
    ref = torch.softmax(act, -1)[:, 0].numpy()
    for j in range(len(c["markers"])):
        k = len(c["markers"][j]); mine = rn.act_probability(head, hq[j, 0].numpy(), lg[j, :k].numpy().astype(np.float64))
        worst = max(worst, abs(mine - ref[j])); n += 1
print(f"questions {n}: max |act_probability numpy - torch| {worst:.2e}")
