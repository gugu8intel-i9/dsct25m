import torch

from dsct.model import DSCT, count_params


def test_forward_backward():
    m = DSCT(vocab=512, d=128, layers=2, heads=4, emb_dim=32, mtp_k=2, deliberate_steps=2)
    x = torch.randint(0, 512, (2, 40))
    out = m(x[:, :-1], targets=x)
    out["loss"].backward()
    assert out["logits"].shape == (2, 39, 512)
    assert torch.isfinite(out["loss"])


def test_gates_start_closed():
    m = DSCT(vocab=512, d=128, layers=2, heads=4, emb_dim=32)
    for layer in m.layers:
        assert torch.tanh(layer.crosstalk.gate_a).item() == 0.0
        assert torch.tanh(layer.crosstalk.gate_b).item() == 0.0


def test_param_budget():
    big = DSCT(vocab=50257, d=384, layers=3, heads=6)
    n = count_params(big)
    print(f"dsct-25m config: {n/1e6:.2f}M params")
    assert n < 26_000_000
