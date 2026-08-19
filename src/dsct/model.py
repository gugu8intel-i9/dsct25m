"""Dual-Stream Cognitive Transformer (DSCT) architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import TernaryLinear


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.register_buffer(
            "inv", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        )

    def forward(self, x, T: int):
        f = torch.outer(torch.arange(T, device=x.device), self.inv)
        cos, sin = f.cos()[None, :, None, :], f.sin()[None, :, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = TernaryLinear(d, 3 * d)
        self.o = TernaryLinear(d, d)
        self.rot = Rotary(self.dk)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = self.rot(q.view(B, T, self.h, self.dk), T)
        k = self.rot(k.view(B, T, self.h, self.dk), T)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.view(B, T, self.h, self.dk).transpose(1, 2),
            is_causal=True,
        )
        return self.o(att.transpose(1, 2).reshape(B, T, D))


class StreamBlock(nn.Module):
    """One cognitive stream: causal self-attention + ternary MLP."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = CausalSelfAttention(d, heads)
        self.mlp = nn.Sequential(
            TernaryLinear(d, 4 * d), nn.GELU(), TernaryLinear(4 * d, d)
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class GatedCrossAttention(nn.Module):
    """Bidirectional communication channel between the two streams.

    Each stream queries the other stream's keys/values causally. Gates are
    initialized at zero so the streams start independent and learn how much
    to communicate during training.
    """

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dk = heads, d // heads
        # Stream A queries stream B
        self.q_a = TernaryLinear(d, d)
        self.kv_b = TernaryLinear(d, 2 * d)
        self.o_a = TernaryLinear(d, d)
        # Stream B queries stream A
        self.q_b = TernaryLinear(d, d)
        self.kv_a = TernaryLinear(d, 2 * d)
        self.o_b = TernaryLinear(d, d)
        self.gate_a = nn.Parameter(torch.zeros(1))
        self.gate_b = nn.Parameter(torch.zeros(1))
        self.na, self.nb = RMSNorm(d), RMSNorm(d)

    def _attend(self, q_src, kv_src, q_proj, kv_proj, o_proj):
        B, T, D = q_src.shape
        q = q_proj(q_src).view(B, T, self.h, self.dk).transpose(1, 2)
        k, v = kv_proj(kv_src).chunk(2, -1)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return o_proj(att.transpose(1, 2).reshape(B, T, D))

    def forward(self, a, b):
        a2 = self._attend(self.na(a), b, self.q_a, self.kv_b, self.o_a)
        b2 = self._attend(self.nb(b), a, self.q_b, self.kv_a, self.o_b)
        return a + torch.tanh(self.gate_a) * a2, b + torch.tanh(self.gate_b) * b2


class DSCTLayer(nn.Module):
    """System 1 (fast/semantic) + System 2 (slow/reasoning) + crosstalk."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.system1 = StreamBlock(d, heads)
        self.system2 = StreamBlock(d, heads)
        self.crosstalk = GatedCrossAttention(d, heads)

    def forward(self, s1, s2):
        s1 = self.system1(s1)
        s2 = self.system2(s2)
        return self.crosstalk(s1, s2)


class DSCT(nn.Module):
    """Dual-Stream Cognitive Transformer language model.

    Args:
        vocab: vocabulary size
        d: model dimension
        layers: number of DSCT layers (each contains both streams)
        heads: attention heads per stream
        emb_dim: factorized embedding dimension (ALBERT-style)
        mtp_k: number of multi-token prediction heads
        deliberate_steps: recurrent refinement steps on System 2
    """

    def __init__(
        self,
        vocab: int = 50257,
        d: int = 384,
        layers: int = 3,
        heads: int = 6,
        emb_dim: int = 128,
        mtp_k: int = 2,
        deliberate_steps: int = 2,
    ):
        super().__init__()
        # Factorized embedding keeps a 50k vocab affordable at 25M params
        self.emb = nn.Embedding(vocab, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, d, bias=False)
        self.unemb_proj = nn.Linear(d, emb_dim, bias=False)

        self.s1_in = TernaryLinear(d, d)
        self.s2_in = TernaryLinear(d, d)
        self.layers = nn.ModuleList([DSCTLayer(d, heads) for _ in range(layers)])

        # Deliberation: shared-weight recurrent refiner on System 2.
        # Runs `deliberate_steps` times -> deeper reasoning, zero extra params.
        self.deliberate_steps = deliberate_steps
        self.reasoner = StreamBlock(d, heads)
        self.halt = nn.Parameter(torch.zeros(1))  # learned "how much to think"

        self.fuse_gate = nn.Parameter(torch.zeros(1))
        self.norm = RMSNorm(d)

        # MTP heads: predict tokens t+2 ... t+mtp_k+1 (parallel variant)
        self.mtp_k = mtp_k
        self.mtp_proj = nn.ModuleList([TernaryLinear(2 * d, d) for _ in range(mtp_k)])
        self.mtp_norm = nn.ModuleList([RMSNorm(d) for _ in range(mtp_k)])

    def _logits(self, h):
        return F.linear(self.unemb_proj(h), self.emb.weight)  # tied embeddings

    def forward(self, idx, targets=None):
        x = self.emb_proj(self.emb(idx))
        s1, s2 = self.s1_in(x), self.s2_in(x)
        for layer in self.layers:
            s1, s2 = layer(s1, s2)

        # Recurrent deliberation on the reasoning stream
        for _ in range(self.deliberate_steps):
            s2 = s2 + torch.tanh(self.halt) * (self.reasoner(s2) - s2)

        # Gated fusion of the two streams
        g = torch.sigmoid(self.fuse_gate)
        h = self.norm(g * s1 + (1 - g) * s2)
        out = {"logits": self._logits(h)}

        if targets is not None:
            logits = out["logits"]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            for k in range(self.mtp_k):
                shift = k + 2
                if targets.shape[1] <= shift:
                    break
                hk = self.mtp_norm[k](
                    self.mtp_proj[k](torch.cat([h[:, :-shift], x[:, :-shift]], -1))
                )
                lk = self._logits(hk)
                loss = loss + 0.3 / self.mtp_k * F.cross_entropy(
                    lk.reshape(-1, lk.size(-1)),
                    targets[:, shift:].reshape(-1),
                    ignore_index=-100,
                )
            out["loss"] = loss
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
