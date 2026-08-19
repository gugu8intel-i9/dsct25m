# DSCT-25M: Dual-Stream Cognitive Transformer

A **25M-parameter**, **ternary (1.58-bit)** language model with a novel dual-stream cognitive architecture, chain-of-thought (CoT) training, multi-token prediction (MTP), and inference-time deliberation about which token to predict next.


## Architecture

```
dsct25m/
├── LICENSE
├── README.md
├── pyproject.toml
├── configs/
│   └── dsct_25m.yaml
├── scripts/
│   ├── train.py
│   ├── train_kaggle.py
│   └── smoke_test.py
├── src/
│   └── dsct/
│       ├── __init__.py
│       ├── quant.py
│       ├── model.py
│       ├── tokenizer.py
│       ├── data.py
│       └── generate.py
└── tests/
    └── test_model.py
```

### Key components

| Component | What it does |
|---|---|
| **Ternary weights** | All linear layers use BitNet b1.58-style absmean quantization to {-1, 0, +1} with a straight-through estimator, plus 8-bit activation quantization. ~4× memory savings vs FP16 at inference. |
| **Dual streams** | System 1 (fast/semantic) and System 2 (slow/reasoning) are separate transformer stacks that process every token in parallel. |
| **Gated cross-attention** | Each stream causally attends to the other. Gates initialize at **zero**, so streams start independent and *learn* how much to communicate — watch the gate values open during training. |
| **Deliberation loop** | A shared-weight recurrent refiner runs R extra steps on System 2 per forward pass: deeper reasoning with zero extra parameters, controlled by a learned halting gate. |
| **MTP heads** | Extra heads predict tokens t+2 and t+3 (DeepSeek-V3 style, parallel variant), improving representation quality and enabling speculative decoding later. |
| **Factorized embeddings** | ALBERT-style factorization (vocab → 128 → d) keeps a 50k vocab affordable at 25M params. Output head is tied to the embedding. |
| **CoT** | Training data is wrapped in `<think>` tags; inference prompts trigger explicit step-by-step reasoning. |
| **Lookahead reranking** | At generation time, the top-k candidate tokens are each scored by the logprob of a short rollout — the model literally deliberates about which token to pick next. |

## Repo layout

```
dsct25m/
├── README.md
├── pyproject.toml
├── configs/
│   └── dsct_25m.yaml      # model + training hyperparameters
├── src/
│   └── dsct/
│       ├── __init__.py
│       ├── quant.py       # ternary quantization (BitNet-style)
│       ├── model.py       # DSCT architecture + MTP heads
│       ├── tokenizer.py   # gigatoken wrapper (HF fallback)
│       ├── data.py        # dataset loading + CoT formatting
│       └── generate.py    # CoT + lookahead decoding
├── scripts/
│   ├── train.py
│   └── smoke_test.py
└── tests/
    └── test_model.py
```

## Install

```bash
pip install -e .          # core (torch, datasets, transformers)
pip install -e ".[fast]"  # + gigatoken (Rust tokenizer, ~1000x faster)
```

## Quickstart

```bash
# Verify the architecture runs (no GPU/datasets needed)
python scripts/smoke_test.py

# Run tests (forward/backward, gate init, param budget)
pytest tests/

# Train
python scripts/train.py
```

### Generate

```python
import torch
from dsct import DSCT, GigaTokenizer, generate

tok = GigaTokenizer("openai-community/gpt2")
model = DSCT(vocab=tok.vocab_size, d=384, layers=3, heads=6).cuda()
model.load_state_dict(torch.load("dsct25m.pt"))
model.eval()

print(generate(model, tok, "Natalia sold clips to 48 friends in April..."))
```

## Tokenizer

Data preprocessing uses [gigatoken](https://github.com/marcelroed/gigatoken), a Rust BPE engine that tokenizes at GB/s (~1000× faster than HuggingFace tokenizers). It wraps an existing vocab (default: GPT-2, 50,257 tokens) and falls back to plain HF tokenizers automatically if not installed.

## Training data

The loader attempts, in order:

1. `Gugu8/English-Extended`
2. `Gugu8/Math-Dataset`
3. `r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation`
4. **Fallback:** `openai/gsm8k` (real CoT math data)


All text is wrapped in `<think>...</think>` tags so the model learns explicit chain-of-thought structure.

## Config

`configs/dsct_25m.yaml`:

```yaml
tokenizer_base: openai-community/gpt2
seq_len: 1024
d: 384
layers: 3            # each layer = both streams + crosstalk
heads: 6
mtp_k: 2
deliberate_steps: 2
batch: 16
lr: 0.0006
steps: 20000
```

This lands at **~23–25M parameters** (≈6.4M factorized embeddings, ≈14.2M dual-stream layers, ≈1.8M shared reasoner, ≈0.6M MTP). `train.py` asserts the budget; bump `layers` or `d` to hit exactly 25M.

## How the "thinking" works

Three distinct mechanisms, cheapest to most expensive:

1. **Deliberation loop (free-ish):** System 2 is refined R times per forward pass through shared weights — more compute per token, no more params.
2. **MTP (training time):** auxiliary heads force the model to build representations that anticipate multiple future tokens.
3. **Lookahead reranking (inference):** each candidate next-token is scored by a short rollout (~`lookahead × rollout`× slower per token — use for hard problems, not bulk generation).

## Roadmap

- [ ] MTP-based speculative decoding (the real inference payoff of MTP)
- [ ] Ablation script: single-stream vs DSCT at equal params
- [ ] W&B logging + eval harness (GSM8K accuracy)
- [ ] Ternary-aware CUDA kernels for actual (not simulated) 1.58-bit speedups

## License

MIT
