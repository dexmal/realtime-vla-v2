from __future__ import annotations
from dataclasses import MISSING
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from pathlib import Path
from typing import Any
from typing import get_type_hints
import yaml

@dataclass
class ClientConfig:
    infer_url: str
    endpoint: str
    timeout_s: float
    run_duration_s: float

@dataclass
class ObserverConfig:
    name: str
    image_size: tuple[int, int]
    fps: int
    state_dim: int
    airbot_host: str
    airbot_left_port: int
    airbot_right_port: int
    top_camera_id: str
    left_camera_id: str
    right_camera_id: str
    enable_cameras: bool

@dataclass
class ExecutorConfig:
    name: str
    enable_init_action: bool = False
    init_action: list[float] = field(default_factory=list)
    init_steps: int = 100
    init_sleep_s: float = 0.01
    control_dt_s: float = 0.02
    obs_image_delay_ms: float = 55.0
    state_delay_s: float = 0.05
    max_prefill_states: int = 15
    heartbeat_history_len: int = 200
    planner_dims: list[int] = field(default_factory=list)
    airbot_host: str = "localhost"
    airbot_left_port: int = 50051
    airbot_right_port: int = 50053
    left_gripper_bias: float = 0.0
    right_gripper_bias: float = 0.0
    infer_fixed_dims: list[int] = field(default_factory=list)
    infer_fixed_values: list[float] = field(default_factory=list)
    command_fixed_dims: list[int] = field(default_factory=list)
    command_fixed_values: list[float] = field(default_factory=list)
    action_interval_ms: float = 10.0
    action_speed_limit_per_s: float = 0.0
    action_speed_limit_dims: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    )
    enable_servo_interpolation: bool = False
    servo_interval_ms: float = 10.0
    savgol_window_length: int = 1
    savgol_polyorder: int = 3
    forward_track_alpha: float = 1.0
    forward_track_delay_cnt: int = 5
    forward_track_lead_s: float = 0.15
    mpc_contact_threshold: float = 0.15
    mpc_contact_hold_min: int = 8
    mpc_tau: float = 0.15
    mpc_horizon_n: int = 15
    mpc_w_track0: float = 10.0
    mpc_track_decay: float = 1.0
    mpc_w_cmd: float = 0.0
    mpc_w_yx: float = 0.0
    mpc_w_dy: float = 0.0
    mpc_w_ddy: float = 20.0
    mpc_contact_v_scale: float = 0.4
    mpc_q_min: list[float] = field(default_factory=list)
    mpc_q_max: list[float] = field(default_factory=list)
    mpc_v_max: list[float] = field(default_factory=list)
    mpc_dy_max: list[float] = field(default_factory=list)
    mpc_ddy_max: list[float] = field(default_factory=list)
    gripper_heartbeat_lookahead_ms: float = 90.0
    gripper_lookahead_dims: list[int] = field(default_factory=lambda: [6, 13])

@dataclass
class VisualizationConfig:
    output_dir: str
    enable_recording: bool
    record_videos: bool = True
    record_rerun: bool = True
    max_pending_video_frames: int = 32

@dataclass
class Config:
    client: ClientConfig
    observer: ObserverConfig
    executor: ExecutorConfig
    visualization: VisualizationConfig

def _dict_to_dataclass(cls: type, data: dict[str, Any], prefix: str = "") -> Any:
    known_names = {f.name for f in fields(cls)}
    extra = [k for k in data if k not in known_names]
    if extra:
        raise KeyError(f"Unknown config keys under {prefix or 'root'}: {extra}")
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for f in fields(cls):
        if f.name not in data:
            if f.default is not MISSING:
                kwargs[f.name] = f.default
                continue
            if f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
                continue
            missing.append(f.name)
            continue
        value = data[f.name]
        field_type = type_hints.get(f.name, f.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[f.name] = _dict_to_dataclass(field_type, value, prefix=f"{prefix}{f.name}.")
        else:
            kwargs[f.name] = value
    if missing:
        raise KeyError(f"Missing config keys under {prefix or 'root'}: {missing}")
    return cls(**kwargs)

def load_config(config_path: str | Path) -> Config:
    with Path(config_path).open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    return _dict_to_dataclass(Config, data)
