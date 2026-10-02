"""Config dataclasses plus YAML inheritance with content hashing.

Every experiment must resolve to a deterministic hash so two runs can be
proved to have used the same configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

CONFIG_DIR = os.environ.get("AGENTMODEL_CONFIG_DIR", "configs")


@dataclass
class RopeScalingConfig:
    factor: float = 1.0
    original_max_seq_len: int = 2048
    type: str = "none"  # none | yarn | linear

    @property
    def active(self) -> bool:
        return self.type != "none" and self.factor != 1.0


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 12
    d_model: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    d_ff: int = 1408
    max_seq_len: int = 2048

    norm_eps: float = 1e-6
    qk_norm: bool = True
    zero_centered_qk_norm: bool = False

    rope_theta: float = 500_000.0
    partial_rope_dim: int = 64
    rope_scaling: RopeScalingConfig = field(default_factory=RopeScalingConfig)

    layer_pattern: str = "swa"  # global | swa | hybrid
    local_layers: str = "swa"  # swa | delta | full
    global_layers: str = "full"
    local_window: int = 512
    swa_every: int = 5

    gated_attention: bool = True
    attn_output_gate: bool = True

    n_expert: int = 0
    n_expert_used: int = 0
    n_shared_expert: int = 1
    moe_aux_loss_coeff: float = 0.001

    tie_embeddings: bool = False
    mtp_n_layer: int = 0
    logit_scale: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.rope_scaling, dict):
            self.rope_scaling = RopeScalingConfig(**self.rope_scaling)
        self.validate()

    def validate(self) -> None:
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        head_dim = self.d_model // self.n_head
        if self.partial_rope_dim > head_dim:
            raise ValueError(
                f"partial_rope_dim ({self.partial_rope_dim}) exceeds head_dim ({head_dim})"
            )
        if self.partial_rope_dim % 2 != 0:
            raise ValueError("partial_rope_dim must be even")
        if self.n_expert and self.n_expert_used == 0:
            raise ValueError("n_expert_used must be > 0 when n_expert > 0")
        if self.n_expert and not 1 <= self.n_expert_used <= self.n_expert:
            raise ValueError("n_expert_used must be in [1, n_expert]")
        if self.layer_pattern not in {"global", "swa", "hybrid"}:
            raise ValueError(f"unknown layer_pattern {self.layer_pattern}")
        if self.local_window > self.max_seq_len:
            raise ValueError("local_window must not exceed max_seq_len")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_kv_groups(self) -> int:
        return self.n_head // self.n_kv_head

    def layer_types(self) -> list[str]:
        """Resolve the per-layer attention type from the layer pattern.

        `swa`    -> one "full" global layer every `swa_every` local layers
        `hybrid` -> one "full" global layer every `swa_every` local layers,
                    but the local layers use linear attention instead
        `global` -> every layer is full attention
        """
        if self.layer_pattern == "global":
            return ["full"] * self.n_layer
        if self.layer_pattern == "hybrid":
            local = "delta"
        else:
            local = self.local_layers
        out: list[str] = []
        for i in range(self.n_layer):
            is_global = (i + 1) % self.swa_every == 0 or i == 0
            out.append(self.global_layers if is_global else local)
        if self.n_layer > 1 and out[-1] != self.global_layers:
            # The final layer always sees the whole context, so long-range
            # information has a direct path into the logits.
            out[-1] = self.global_layers
        return out

    def _attn_params(self, layer_type: str) -> int:
        d, h, hd, kv = self.d_model, self.n_head, self.head_dim, self.n_kv_head
        if layer_type == "delta":
            # q (n_head), k (n_kv_head), v (n_head), o (n_head) + b/a gates
            return 3 * h * hd * d + kv * hd * d + 2 * h
        params = h * hd * d + 2 * kv * hd * d  # q, k, v
        params += d * d  # out projection
        if self.attn_output_gate and self.gated_attention:
            params += d * d
        if self.qk_norm:
            params += 2 * hd  # q_norm and k_norm gains
        return params

    @property
    def n_attn_params_layer(self) -> int:
        return self._attn_params("full")

    def n_params(self, non_embedding: bool = False) -> int:
        d = self.d_model
        attn = sum(self._attn_params(lt) for lt in self.layer_types())
        if self.n_expert:
            per_expert = 3 * self.d_ff * d
            ffn = self.n_expert * per_expert + (self.n_shared_expert * per_expert)
            ffn += d * self.n_expert  # router
            if self.layer_types().count("delta"):
                ffn += self.layer_types().count("delta") * 2 * self.n_head  # delta b/a gates
        else:
            ffn = 3 * self.d_ff * d * self.n_layer
        norm = 2 * d * self.n_layer + d  # per-block pair + final norm
        head = d * self.vocab_size
        emb = self.vocab_size * d
        total = attn + ffn + norm + head + emb
        if self.mtp_n_layer:
            total += self.mtp_n_layer * (attn // self.n_layer + ffn // self.n_layer + norm)
            total += self.mtp_n_layer * self.vocab_size * d
        if self.tie_embeddings:
            total -= d * self.vocab_size
        if non_embedding:
            total -= self.vocab_size * d
        return total


@dataclass
class DataConfig:
    sequence_length: int = 2048
    eos_token_id: int = 2
    bos_token_id: int = 1
    pad_token_id: int = 0


@dataclass
class OptimConfig:
    optimizer: str = "muon_adamw"
    lr: float = 3e-3
    muon_lr: float = 0.02
    momentum: float = 0.95
    nesterov: bool = True
    weight_decay: float = 0.1
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    decay_steps: int = 20_000
    schedule: str = "wsd"
    min_lr_ratio: float = 0.1


@dataclass
class PrecisionConfig:
    compute_dtype: str = "bf16"
    param_dtype: str = "fp32"
    ema_decay: float = 0.0


@dataclass
class TrainConfig:
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    seed: int = 1234
    log_interval: int = 10
    checkpoint_interval: int = 1000
    gradient_checkpointing: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "optim": OptimConfig,
    "precision": PrecisionConfig,
    "train": TrainConfig,
}


def _load_yaml_with_inheritance(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    extends = raw.pop("extends", None)
    if extends:
        parent_path = extends if os.path.isabs(extends) else os.path.join(os.path.dirname(path), extends)
        parent = _load_yaml_with_inheritance(parent_path)
        raw = deep_merge(parent, raw)
    return raw


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load a YAML config, resolving `extends` chains relative to the file."""
    path = os.fspath(path)
    raw = _load_yaml_with_inheritance(path)
    raw.setdefault("meta", {})
    raw["meta"]["source"] = os.path.basename(path)
    kwargs: dict[str, Any] = {}
    for section, klass in _SECTIONS.items():
        if section in raw:
            kwargs[section] = klass(**raw.pop(section))
    raw.pop("extends", None)
    kwargs["meta"] = raw.pop("meta", {})
    return Config(**kwargs)


def config_from_dict(d: dict[str, Any]) -> Config:
    kwargs: dict[str, Any] = {}
    for section, klass in _SECTIONS.items():
        if section in d and d[section] is not None:
            kwargs[section] = klass(**d[section])
    return Config(meta=d.get("meta", {}), **kwargs)


def config_hash_of(cfg: Config) -> str:
    return cfg.config_hash()
