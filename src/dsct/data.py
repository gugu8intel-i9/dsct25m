"""Dataset loading and chain-of-thought formatting."""

from datasets import load_dataset, interleave_datasets

DATASETS = [
    "Gugu8/English-Extended",
    "Gugu8/Math-Dataset",
    "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation",
]
FALLBACKS = ["openai/gsm8k"]  # real CoT math data if the above fail to load


def format_cot(text: str) -> str:
    """Wrap reasoning in <think> tags so the model learns explicit CoT."""
    if "<think>" in text:
        return text
    return f"<think>\n{text}\n</think>"


def build_dataset(tok, seq_len: int = 1024):
    dsets = []
    for name in DATASETS + FALLBACKS:
        try:
            dsets.append(load_dataset(name, split="train"))
            print(f"[data] loaded {name}")
            if len(dsets) == 1 and name in FALLBACKS:
                break
        except Exception as e:
            print(f"[data] skipping {name}: {type(e).__name__}")
    if not dsets:
        raise RuntimeError("No datasets could be loaded.")
    ds = interleave_datasets(dsets) if len(dsets) > 1 else dsets[0]

    def tok_fn(ex):
        text = (
            ex.get("text")
            or ex.get("content")
            or (
                f"Q: {ex['question']}\nA: {ex['answer']}"
                if "question" in ex
                else str(ex)
            )
        )
        return {"input_ids": tok.encode(format_cot(text), max_length=seq_len)}

    return ds.map(tok_fn, remove_columns=ds.column_names)
