"""Inference: CoT prompting + lookahead reranking.

For each step, the top-k candidate tokens are scored by the model's average
logprob over a short rollout -- the model literally deliberates about which
token to predict next.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    tok,
    prompt: str,
    max_new: int = 256,
    cot: bool = True,
    lookahead: int = 4,
    rollout: int = 4,
    temp: float = 0.8,
    device: str = "cuda",
) -> str:
    if cot:
        prompt = f"Q: {prompt}\nA: <think>\n"
    ids = torch.tensor([tok.encode(prompt)], device=device)
    for _ in range(max_new):
        logits = model(ids)["logits"][:, -1]
        topk = torch.topk(logits, lookahead)
        best, best_score = None, float("-inf")
        for cand, cand_lp in zip(topk.indices[0], topk.values[0]):
            s = torch.cat([ids, cand.view(1, 1)], 1)
            score = cand_lp.item()
            for _ in range(rollout):
                l = model(s)["logits"][:, -1]
                nxt = torch.multinomial(F.softmax(l / temp, -1), 1)
                score += torch.log_softmax(l, -1)[0, nxt.item()].item()
                s = torch.cat([s, nxt], 1)
            if score > best_score:
                best, best_score = cand, score
        ids = torch.cat([ids, best.view(1, 1)], 1)
        if best.item() == tok.eos_id:
            break
    return tok.decode(ids[0].tolist())
