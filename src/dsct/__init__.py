"""DSCT-25M: Dual-Stream Cognitive Transformer.

A 25M-parameter ternary (1.58-bit) language model with:
- Dual cognitive streams (System 1 / System 2) with gated cross-attention
- Recurrent deliberation on the reasoning stream
- Multi-token prediction (MTP) heads
- Chain-of-thought training and inference
"""

from .model import DSCT, count_params
from .tokenizer import GigaTokenizer
from .generate import generate

__version__ = "0.1.0"
__all__ = ["DSCT", "count_params", "GigaTokenizer", "generate", "__version__"]
