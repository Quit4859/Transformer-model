import torch

from agentmodel.config import load_config
from agentmodel.scale.quantization import (
    dequantize_int8,
    export_state_dict,
    quantize_int8,
)


def test_int8_quantization_is_bounded_and_approximately_reversible():
    source = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    quantized = quantize_int8(source)
    assert quantized.values.dtype == torch.int8
    assert quantized.values.abs().max().item() <= 127
    assert torch.allclose(dequantize_int8(quantized), source, atol=1 / 127)


def test_zero_tensor_has_finite_scale():
    quantized = quantize_int8(torch.zeros(3))
    assert torch.isfinite(quantized.scale)
    assert torch.equal(dequantize_int8(quantized), torch.zeros(3))


def test_quantization_rejects_non_float_and_wrong_dtype():
    for bad in (torch.ones(3, dtype=torch.int32),):
        try:
            quantize_int8(bad)
        except TypeError:
            pass
        else:
            raise AssertionError("expected TypeError for non-floating input")

    from agentmodel.scale.quantization import QuantizedTensor

    try:
        dequantize_int8(QuantizedTensor(values=torch.ones(3), scale=torch.ones(3)))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-int8 values")


def test_real_model_weights_survive_a_quantize_dequantize_round_trip(tmp_path):
    from agentmodel.model.transformer import Transformer

    cfg = load_config("configs/pretrain_nano.yaml").model
    model = Transformer(cfg)
    payload = {}
    for name, tensor in model.state_dict().items():
        if not tensor.is_floating_point():
            continue
        restored = dequantize_int8(quantize_int8(tensor))
        error = (restored - tensor).abs().max().item()
        assert error <= 2 * tensor.abs().max().item() / 127 + 1e-6, name
        payload[name] = restored
    assert payload


def test_export_state_dict_writes_quantized_payload(tmp_path):
    path = tmp_path / "model.int8.pt"
    export_state_dict({"w": torch.randn(4, 4), "step": torch.tensor(3)}, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["w"]["values"].dtype == torch.int8
    assert loaded["w"]["scale"] is not None
    assert loaded["step"]["scale"] is None
    assert torch.equal(loaded["step"]["values"], torch.tensor(3))