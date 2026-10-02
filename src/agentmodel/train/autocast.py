"""Autocast context that degrades gracefully on CPU (no bf16 kernels required)."""

from __future__ import annotations

import contextlib

import torch

from .precision import resolve_dtype


def autocast_ctx(compute_dtype: str, enabled: bool = True, device_type: str | None = None):
    dtype = resolve_dtype(compute_dtype)
    device_type = device_type or ("cuda" if torch.cuda.is_available() else "cpu")
    if not enabled or device_type == "cpu" and dtype == torch.float8_e4m3fn:
        return contextlib.nullcontext()
    if device_type == "cpu" and dtype not in (torch.bfloat16, torch.float32):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)
