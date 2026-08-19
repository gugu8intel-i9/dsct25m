import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from dsct.model import DSCT, count_params

m = DSCT(vocab=512, d=128, layers=2, heads=4, emb_dim=32, mtp_k=2, deliberate_steps=2)
print(f"params (toy config): {count_params(m):,}")
x = torch.randint(0, 512, (2, 40))
out = m(x[:, :-1], targets=x)
out["loss"].backward()
assert out["logits"].shape == (2, 39, 512)
assert torch.tanh(m.layers[0].crosstalk.gate_a).item() == 0.0  # gates start closed
print("loss:", out["loss"].item())
print("OK")

big = DSCT(vocab=50257, d=384, layers=3, heads=6)
print(f"params (dsct-25m config): {count_params(big)/1e6:.2f}M")
