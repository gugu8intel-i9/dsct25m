"""DSCT-25M training script.

Fully driven by configs/dsct_25m.yaml:
  - linear LR warmup + cosine decay
  - gradient accumulation
  - optional fp16 autocast + grad scaling (for T4-class GPUs)
  - periodic checkpointing to out_dir
"""

import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
import yaml
from torch.utils.data import DataLoader

from dsct.model import DSCT, count_params
from dsct.tokenizer import GigaTokenizer
from dsct.data import build_dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs" / "dsct_25m.yaml"))

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = cfg.get("fp16", False) and device == "cuda"


# ---------------- LR schedule: linear warmup -> cosine decay ----------------
def lr_at(step: int) -> float:
    base = cfg["lr"]
    warm = cfg.get("warmup_steps", 0)
    total = cfg["steps"]
    if step < warm:
        return base * (step + 1) / max(1, warm)
    # cosine decay to 10% of peak
    t = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


# ---------------- Data ----------------
tok = GigaTokenizer(cfg["tokenizer_base"])
ds = build_dataset(tok, cfg["seq_len"])


def collate(batch):
    seqs = [torch.tensor(b["input_ids"][: cfg["seq_len"]]) for b in batch]
    mx = max(len(s) for s in seqs)
    x = torch.full((len(seqs), mx), tok.pad_id)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = s
    return x


dl = DataLoader(
    ds.shuffle(seed=0),
    batch_size=cfg["batch"],
    collate_fn=collate,
    num_workers=2 if device == "cuda" else 0,
    pin_memory=device == "cuda",
)

# ---------------- Model ----------------
model = DSCT(
    vocab=tok.vocab_size,
    d=cfg["d"],
    layers=cfg["layers"],
    heads=cfg["heads"],
    emb_dim=cfg.get("emb_dim", 128),
    mtp_k=cfg.get("mtp_k", 2),
    deliberate_steps=cfg.get("deliberate_steps", 2),
).to(device)

n = count_params(model)
print(f"Params: {n/1e6:.2f}M")
assert n < 26_000_000, "over budget - reduce layers or d"

opt = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["lr"],
    weight_decay=cfg.get("weight_decay", 0.1),
)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
accum = cfg.get("grad_accum", 1)
clip = cfg.get("grad_clip", 1.0)

out_dir = ROOT / cfg.get("out_dir", "checkpoints")
out_dir.mkdir(exist_ok=True)
save_every = cfg.get("save_every", 2000)

# ---------------- Train ----------------
model.train()
t0 = time.time()
opt.zero_grad(set_to_none=True)

for step, batch in enumerate(dl):
    batch = batch.to(device, non_blocking=True)

    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr

    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        out = model(batch[:, :-1], targets=batch)
        loss = out["loss"] / accum

    scaler.scale(loss).backward()

    if (step + 1) % accum == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    if step % 100 == 0:
        ga = torch.tanh(model.layers[-1].crosstalk.gate_a).item()
        gb = torch.tanh(model.layers[-1].crosstalk.gate_b).item()
        mem = (
            f" | mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB"
            if device == "cuda"
            else ""
        )
        print(
            f"step {step:6d} | loss {out['loss'].item():.4f} | lr {lr:.2e} | "
            f"gates {ga:.3f}/{gb:.3f} | {time.time()-t0:.0f}s{mem}"
        )

    if step > 0 and step % save_every == 0:
        path = out_dir / f"dsct25m_step{step}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "cfg": cfg,
            },
            path,
        )
        print(f"saved {path}")

    if step >= cfg["steps"]:
        break

final = out_dir / "dsct25m_final.pt"
torch.save(
    {"model": model.state_dict(), "opt": opt.state_dict(), "step": cfg["steps"], "cfg": cfg},
    final,
)
print(f"done - saved {final}")
