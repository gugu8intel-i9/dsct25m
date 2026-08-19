import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from torch.utils.data import DataLoader

from dsct.model import DSCT, count_params
from dsct.tokenizer import GigaTokenizer
from dsct.data import build_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs" / "dsct_25m.yaml"))

tok = GigaTokenizer(cfg["tokenizer_base"])
ds = build_dataset(tok, cfg["seq_len"])


def collate(batch):
    seqs = [torch.tensor(b["input_ids"][: cfg["seq_len"]]) for b in batch]
    mx = max(len(s) for s in seqs)
    x = torch.full((len(seqs), mx), tok.pad_id)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = s
    return x


dl = DataLoader(ds.shuffle(seed=0), batch_size=cfg["batch"], collate_fn=collate)
model = DSCT(
    vocab=tok.vocab_size,
    d=cfg["d"],
    layers=cfg["layers"],
    heads=cfg["heads"],
    mtp_k=cfg["mtp_k"],
    deliberate_steps=cfg["deliberate_steps"],
).cuda()
n = count_params(model)
print(f"Params: {n/1e6:.2f}M")
assert n < 26_000_000, "over budget - reduce layers or d"

opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.1)
model.train()
for step, batch in enumerate(dl):
    batch = batch.cuda()
    out = model(batch[:, :-1], targets=batch)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 100 == 0:
        ga = torch.tanh(model.layers[-1].crosstalk.gate_a).item()
        gb = torch.tanh(model.layers[-1].crosstalk.gate_b).item()
        print(f"step {step} | loss {out['loss'].item():.4f} | gates a/b: {ga:.3f}/{gb:.3f}")
    if step >= cfg["steps"]:
        break
torch.save(model.state_dict(), ROOT / "dsct25m.pt")
