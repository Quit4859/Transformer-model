"""Top-level config re-exports so `agentmodel.config` works from anywhere."""

from .model.config import (
    Config,
    DataConfig,
    ModelConfig,
    OptimConfig,
    PrecisionConfig,
    TrainConfig,
    config_from_dict,
    config_hash_of,
    deep_merge,
    load_config,
)

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "PrecisionConfig",
    "TrainConfig",
    "load_config",
    "config_from_dict",
    "config_hash_of",
    "deep_merge",
]
