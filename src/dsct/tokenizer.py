"""Gigatoken-backed tokenizer with HuggingFace fallback.

Gigatoken is a Rust BPE engine (~1000x faster than HF tokenizers) that wraps
an existing vocab. If it isn't installed (pip install dsct25m[fast]), we fall
back to HF tokenizers transparently.
"""


class GigaTokenizer:
    def __init__(self, base: str = "openai-community/gpt2"):
        self.backend = "hf"
        try:
            import gigatoken as gt
            from transformers import AutoTokenizer

            self.tok = gt.Tokenizer(AutoTokenizer.from_pretrained(base)).as_hf()
            self.backend = "gigatoken"
        except Exception as e:
            print(f"[tokenizer] gigatoken unavailable ({e}); using HF tokenizers")
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        return self.tok(
            text, truncation=max_length is not None, max_length=max_length
        )["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id

    @property
    def eos_id(self) -> int:
        return self.tok.eos_token_id
