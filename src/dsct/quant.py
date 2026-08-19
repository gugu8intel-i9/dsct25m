"""Ternary (1.58-bit) weight quantization, BitNet b1.58 style."""

import torch
import torch.nn as nn


class TernaryQuant(torch.autograd.Function):
    """Absmean ternary quantization with straight-through estimator.

    Weights are quantized to {-1, 0, +1} scaled by the mean absolute value.
    Gradients flow through unchanged (STE) so training happens in full precision.
    """

    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean().clamp(min=1e-5)
        return torch.clamp(torch.round(w / scale), -1, 1) * scale

    @staticmethod
    def backward(ctx, g):
        return g


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights and 8-bit quantized activations."""

    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        # 8-bit per-token activation quantization (STE)
        s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        xq = torch.clamp(torch.round(x * s), -127, 127) / s
        xq = x + (xq - x).detach()
        return nn.functional.linear(xq, TernaryQuant.apply(self.weight), self.bias)
