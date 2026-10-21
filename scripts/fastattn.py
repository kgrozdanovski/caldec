"""Exact block-sparse attention for UltraBERT's local layers.

UltraBERT runs two of every three layers with a 128-token window (|i - j| <= 128)
and the third with global attention. Its `sdpa` path builds a dense L x L mask per
window size and runs full attention under it, so the windows save nothing: an 8K
input costs about as much as a dense encoder. `enable(encoder)` swaps in

- an attention function that, for a windowed layer, splits the sequence into
  128-token query blocks and attends each to its own and its two neighbouring key
  blocks (384 keys) under the same |i - j| <= 128 rule, and for a global layer runs
  sdpa with a key-padding mask only; and
- an encoder forward that builds each layer kind's mask once: the blocked mask, a
  key-padding mask for global layers, or (under BLOCK_MIN windows of length, where
  blocking costs more than it saves) one dense windowed mask.

Valid positions see exactly the keys they saw before. A padded query position also
attends to itself, so no row is fully masked (which would put NaN into later
layers); padded positions are never read by the decision head. `check` compares
the two paths on real inputs.
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

NAME = "caldec_local"
# Below BLOCK_MIN windows of length, the blocked layout costs more than it saves.
BLOCK_MIN = 8


def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    return x if n == 1 else x.repeat_interleave(n, dim=1)


def _block_mask(key_valid, window):
    """Mask for the blocked layout: query block b against key blocks b-1, b, b+1."""
    bs, L = key_valid.shape
    B = window
    nb = math.ceil(L / B)
    valid = F.pad(key_valid, (B, nb * B - L + B), value=False).unfold(1, 3 * B, B)   # bs, nb, 3B
    i = torch.arange(B, device=key_valid.device)[:, None]
    j = torch.arange(3 * B, device=key_valid.device)[None, :] - B
    mask = ((j - i).abs() <= window)[None, None] & valid[:, :, None, :]
    return (mask | (j == i)[None, None]).reshape(bs * nb, 1, B, 3 * B)


def _dense_mask(key_valid, window):
    L = key_valid.size(1)
    pos = torch.arange(L, device=key_valid.device)
    near = (pos[:, None] - pos[None, :]).abs() <= window if window else torch.ones(L, L, dtype=torch.bool, device=key_valid.device)
    return (near[None] & key_valid[:, None, :]) | torch.eye(L, dtype=torch.bool, device=key_valid.device)[None]


def _local(q, k, v, mask, window, scaling, dropout):
    bs, h, L, d = q.shape
    B = window
    nb = mask.size(0) // bs
    pad = nb * B - L
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    # SDPA's fused kernels take 4-D inputs only, so fold the blocks into the batch.
    qb = q.view(bs, h, nb, B, d).transpose(1, 2).reshape(bs * nb, h, B, d)
    kp, vp = (F.pad(t, (0, 0, B, B)) for t in (k, v))
    kb = kp.unfold(2, 3 * B, B).permute(0, 2, 1, 4, 3).reshape(bs * nb, h, 3 * B, d)
    vb = vp.unfold(2, 3 * B, B).permute(0, 2, 1, 4, 3).reshape(bs * nb, h, 3 * B, d)
    out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, dropout_p=dropout, scale=scaling)
    return out.view(bs, nb, h, B, d).transpose(1, 2).reshape(bs, h, nb * B, d)[:, :, :L]


def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None, **kwargs):
    """`attention_mask` here is the per-window mask `_forward` built: ("dense", B x 1 x L x L),
    ("block", B*nb x 1 x 128 x 384) or ("global", B x 1 x 1 x L)."""
    n = getattr(module, "num_key_value_groups", 1)
    key, value = _repeat_kv(key, n), _repeat_kv(value, n)
    kind, mask = attention_mask[sliding_window]
    if kind == "block":
        out = _local(query, key, value, mask, int(sliding_window), scaling, dropout)
    else:
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling)
    return out.transpose(1, 2).contiguous(), None


def _forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
    from transformers.modeling_outputs import BaseModelOutput

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0)
    if attention_mask is None:
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
    key_valid = attention_mask.bool()
    L = key_valid.size(1)
    masks = {}
    for w in set(self.config.attention_windows):
        if w is None:
            masks[w] = ("global", key_valid[:, None, None, :])
        elif L <= BLOCK_MIN * w:     # short input: one small dense windowed mask is cheaper
            masks[w] = ("dense", _dense_mask(key_valid, w)[:, None])
        else:
            masks[w] = ("block", _block_mask(key_valid, w))
    hidden, rope = inputs_embeds, {}
    for base in set(self.config.rope_theta):
        rope[base] = self.rotary_emb[str(base).replace(".", "_")](hidden, position_ids)
    for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        hidden = layer(hidden, attention_mask=masks, position_ids=position_ids,
                       position_embeddings=rope[self.config.rope_theta[i]],
                       sliding_window=self.config.attention_windows[i])
    return BaseModelOutput(last_hidden_state=self.norm(hidden))


def register() -> None:
    from transformers import AttentionInterface

    AttentionInterface.register(NAME, attention)


def enable(encoder) -> bool:
    """Switch an UltraBERT encoder to the block-sparse path; other encoders are left alone."""
    if getattr(encoder.config, "model_type", None) != "ultrabert":
        return False
    register()
    encoder.config._attn_implementation = NAME
    for module in encoder.modules():
        if hasattr(module, "config") and module is not encoder:
            module.config._attn_implementation = NAME
    encoder.forward = types.MethodType(_forward, encoder)
    return True
