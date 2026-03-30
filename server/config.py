from __future__ import annotations
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import get_type_hints
import yaml

@dataclass
class ServerConfig:
    host: str
    port: int
    endpoint: str

@dataclass
class ModelConfig:
    adapter: str
    config_name: str
    checkpoint: str
    prompt: str
    adarms_knob: int
    valid_action_num: int
    action_horizon: int
    action_type: str
    image_size: tuple[int, int]
    tokenizer_path: str
    norm_stats_dir: str
    discrete_state_input: bool
    state_dim: int
    action_dim: int
    noise_seed: int | None

@dataclass
class InferenceConfig:
    optimizer: str
    timeaxis_dt_ref_s: float
    timeaxis_dt_min_s: float
    timeaxis_dt_max_s: float
    timeaxis_lambda_acc: float
    timeaxis_lambda_time: float
    timeaxis_stride: int
    timeaxis_optdims: list[int]
    timeaxis_v_max: float | None
    timeaxis_lambda_v: float
    timeaxis_horizon: int
    timeaxis_logging: bool

@dataclass
class Config:
    server: ServerConfig
    model: ModelConfig
    inference: InferenceConfig

def _dict_to_dataclass(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    known_names = {f.name for f in fields(cls)}
    missing = [f.name for f in fields(cls) if f.name not in data]
    if missing:
        raise KeyError(f"Missing config keys under {prefix or 'root'}: {missing}")
    extra = [k for k in data if k not in known_names]
    if extra:
        print(f"[config] ignoring unknown keys under {prefix or 'root'}: {extra}")
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        value = data[f.name]
        field_type = type_hints.get(f.name, f.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, value, prefix=f"{prefix}{f.name}.")
        else:
            kwargs[f.name] = value
    return cls(**kwargs)

def load_config(config_path: str | Path) -> Config:
    with Path(config_path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return _dict_to_dataclass(Config, data)
