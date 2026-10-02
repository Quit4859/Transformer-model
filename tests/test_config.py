import os

import pytest
import yaml

from agentmodel.config import Config, ModelConfig, load_config
from agentmodel.model.transformer import Transformer


def test_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert isinstance(cfg, Config)
    assert cfg.model.n_head % cfg.model.n_kv_head == 0


def test_inheritance_overrides_only_named_fields():
    cfg = load_config("configs/pretrain_nano.yaml")
    base = load_config("configs/base.yaml")
    assert cfg.model.n_layer == 6
    assert cfg.model.d_model == 128
    # untouched fields fall through to the base
    assert cfg.model.norm_eps == base.model.norm_eps
    assert cfg.optim.schedule == base.optim.schedule


def test_config_hash_is_stable_and_sensitive():
    a = load_config("configs/pretrain_nano.yaml")
    b = load_config("configs/pretrain_nano.yaml")
    assert a.config_hash() == b.config_hash()
    b.model.n_layer += 1
    assert a.config_hash() != b.config_hash()


def test_nano_config_matches_actual_param_count():
    cfg = load_config("configs/pretrain_nano.yaml")
    model = Transformer(cfg.model)
    assert abs(model.num_params() - cfg.model.n_params()) < cfg.model.d_model


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_head": 9, "n_kv_head": 4},
        {"d_model": 100, "n_head": 8},
        {"partial_rope_dim": 9999},
        {"partial_rope_dim": 7},
        {"n_expert": 4, "n_expert_used": 0},
        {"n_expert": 4, "n_expert_used": 9},
        {"layer_pattern": "nope"},
        {"local_window": 99999},
    ],
)
def test_invalid_configs_are_rejected(kwargs):
    base = dict(
        vocab_size=64, n_layer=2, d_model=32, n_head=4, n_kv_head=2,
        d_ff=88, max_seq_len=64, partial_rope_dim=8, local_window=16,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        ModelConfig(**base)


def test_yaml_files_all_extend_base():
    for name in os.listdir("configs"):
        if name in {"base.yaml"} or not name.endswith(".yaml"):
            continue
        with open(os.path.join("configs", name)) as f:
            raw = yaml.safe_load(f)
        assert "extends" in raw, f"{name} must declare `extends: base.yaml`"


def test_every_config_is_instantiable():
    for name in sorted(os.listdir("configs")):
        if not name.endswith(".yaml"):
            continue
        cfg = load_config(os.path.join("configs", name))
        assert cfg.model.vocab_size > 0
        assert len(cfg.model.layer_types()) == cfg.model.n_layer