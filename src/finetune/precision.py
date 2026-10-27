"""Choose bf16 vs fp16 for the local GPU.

Turing (RTX 20xx) does not support bf16.  The training scripts must use fp16
there instead of hard-coding bf16.
"""
from __future__ import annotations

from typing import Tuple


def detect_precision() -> Tuple[str, object]:
    """Return (name, torch dtype).  Prefer bf16 only when the GPU supports it."""
    import torch

    if not torch.cuda.is_available():
        return "fp32", torch.float32
    if torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16
