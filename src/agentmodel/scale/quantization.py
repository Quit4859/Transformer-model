"""Simple symmetric int8 quantization for CPU-compatible model export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scale: torch.Tensor


def quantize_int8(tensor: torch.Tensor) -> QuantizedTensor:
    if not tensor.is_floating_point():
        raise TypeError("quantization requires a floating-point tensor")
    maximum = tensor.detach().abs().max()
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127)
    return QuantizedTensor(torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8), scale)


def dequantize_int8(tensor: QuantizedTensor) -> torch.Tensor:
    if tensor.values.dtype != torch.int8:
        raise TypeError("quantized values must have int8 dtype")
    return tensor.values.float() * tensor.scale


def export_state_dict(state_dict: dict[str, torch.Tensor], path: str | Path) -> None:
    """Export floating tensors as int8 plus scales; retain non-floating tensors."""
    payload = {}
    for name, tensor in state_dict.items():
        if tensor.is_floating_point():
            quantized = quantize_int8(tensor.cpu())
            payload[name] = {"values": quantized.values, "scale": quantized.scale}
        else:
            payload[name] = {"values": tensor.cpu(), "scale": None}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
