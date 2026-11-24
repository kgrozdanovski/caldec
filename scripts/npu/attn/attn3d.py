"""Attention formulations that keep the scores in the exMatMul layout. Usage: attn3d.py <variant>"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from rknn.api import RKNN
S, H, D, E = 256, 6, 128, 768
class Attn(nn.Module):
    def __init__(s, kind):
        super().__init__(); torch.manual_seed(0); s.kind = kind
        s.q = nn.Linear(E, E); s.k = nn.Linear(E, E); s.v = nn.Linear(E, E); s.o = nn.Linear(E, E)
    def forward(s, x, mask):                       # x [1,S,E], mask [S,S] additive
        q = s.q(x)[0].view(S, H, D).transpose(0, 1) / D ** 0.5      # [H,S,D]
        k = s.k(x)[0].view(S, H, D).transpose(0, 1)
        v = s.v(x)[0].view(S, H, D).transpose(0, 1)
        sc = torch.bmm(q, k.transpose(1, 2)) + mask                 # [H,S,S]
        kd = s.kind
        if kd.startswith("T"):        # transposed scores [H, key, query]: reductions run over the channel axis
            scT = torch.bmm(k, q.transpose(1, 2)) + mask.t()
            if "mpool" in kd: m = F.max_pool2d(scT.unsqueeze(0), (S, 1))[0]
            elif "tree" in kd:
                m = scT
                while m.shape[1] > 1:
                    h = m.shape[1] // 2; m = torch.maximum(m[:, :h], m[:, h:])
            else: m = scT.amax(1, keepdim=True)                     # [H,1,S]
            e = F.elu(scT - m) + 1.0
            o1 = torch.bmm(torch.cat([v, torch.ones(H, S, 1)], -1).transpose(1, 2), e)   # [H,D+1,Sq]
            oT = o1[:, :D] / o1[:, D:]
            return s.o(oT.permute(2, 0, 1).reshape(1, S, E))
        if kd == "sm3d":
            o = torch.bmm(torch.softmax(sc, -1), v)
        else:
            if kd.startswith("nosm"):
                return s.o(torch.bmm(sc, v).transpose(0, 1).reshape(1, S, E))
            if kd.startswith("rmax"): m = sc.amax(-1, keepdim=True)
            elif kd.startswith("nomax"):     # timing only: no max subtraction
                m = torch.zeros(H, S, 1)
            elif kd.startswith("nosm"):      # timing only: no softmax at all
                return s.o(torch.bmm(sc, v).transpose(0, 1).reshape(1, S, E))
            elif kd.startswith("g16"):      # keys as a 16x16 grid: global pools over (16,16)
                m = F.max_pool2d(sc.view(H, S, 16, 16), (16, 16)).view(H, S, 1)
                e = F.elu(sc - m) + 1.0
                return s.o(torch.bmm(e / (F.avg_pool2d(e.view(H, S, 16, 16), (16, 16)).view(H, S, 1) * S), v).transpose(0, 1).reshape(1, S, E))
            elif kd.startswith("mp4"):      # native exMatMul layout (N=H, C=Sq, H=1, W=Sk)
                m = F.max_pool2d(sc.view(H, S, 1, S), (1, S)).view(H, S, 1)
            elif kd.startswith("relutree"):  # max(a,b) = b + relu(a-b)
                m = sc
                while m.shape[-1] > 1:
                    h = m.shape[-1] // 2; a, b = m[..., :h], m[..., h:]; m = b + F.relu(a - b)
            elif kd.startswith("tree"):
                m = sc
                while m.shape[-1] > 1:
                    h = m.shape[-1] // 2; m = torch.maximum(m[..., :h], m[..., h:])
            else: m = F.max_pool2d(sc.unsqueeze(0), (1, S))[0]       # [H,S,1]
            e = F.elu(sc - m) + 1.0
            if kd.endswith("v1"):     # row sum from the P.V matmul: V' = [V | 1]
                o1 = torch.bmm(e, torch.cat([v, torch.ones(H, S, 1)], -1))
                o = o1[..., :D] / o1[..., D:]
            else:
                o = torch.bmm(e, v) / e.sum(-1, keepdim=True)
        return s.o(o.transpose(0, 1).reshape(1, S, E))
var = sys.argv[1]
m = Attn(var).eval(); x = torch.randn(1, S, E); mask = torch.zeros(S, S)
with torch.no_grad():
    ref = Attn("sm3d").eval()(x, mask); got = m(x, mask); print("torch max|diff| vs softmax:", float((ref - got).abs().max()))
torch.onnx.export(m, (x, mask), f"{var}.onnx", opset_version=17, dynamo=False, input_names=["x", "mask"], output_names=["y"])
r = RKNN(verbose=True); r.config(target_platform="rk3566"); r.load_onnx(f"{var}.onnx"); r.build(do_quantization=False); r.export_rknn(f"{var}.rknn"); r.release()
x.numpy().tofile("0_x.bin"); mask.numpy().tofile("0_mask.bin")
