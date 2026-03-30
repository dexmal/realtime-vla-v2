from __future__ import annotations

from config import Config
from executor import BaseExecutor
from executor import OnDeviceMpcExecutor
from executor import RawActionExecutor
from robot_io import AirbotActuator
from robot_io import AirbotRealObserver
from robot_io import BaseActuator
from robot_io import BaseObserver
from robot_io import MockStateObserver
from robot_io import NoopActuator
from robot_io import create_airbot_robot

def _build_shared_airbot_robot(cfg: Config):
    if cfg.observer.name != "airbot_real":
        return None
    obs = cfg.observer
    ex = cfg.executor
    if (
        obs.airbot_host != ex.airbot_host
        or int(obs.airbot_left_port) != int(ex.airbot_left_port)
        or int(obs.airbot_right_port) != int(ex.airbot_right_port)
    ):
        raise ValueError(
            "observer/executor airbot endpoint mismatch; MPC runtime requires a shared robot connection"
        )
    robot = create_airbot_robot(
        airbot_host=obs.airbot_host,
        left_port=int(obs.airbot_left_port),
        right_port=int(obs.airbot_right_port),
    )
    robot.connect()
    return robot

def build_observer(cfg: Config, pending_actions_provider, shared_robot=None) -> BaseObserver:
    observer_map = {
        "mock": MockStateObserver,
        "airbot_real": AirbotRealObserver,
    }
    observer_cls = observer_map.get(cfg.observer.name)
    if observer_cls is None:
        raise ValueError(f"Unsupported observer.name={cfg.observer.name!r}")
    if observer_cls is AirbotRealObserver:
        return observer_cls.from_config(cfg, pending_actions_provider, robot=shared_robot)
    return observer_cls.from_config(cfg, pending_actions_provider)

def _build_actuator(cfg: Config, shared_robot=None) -> BaseActuator:
    observer_cls = {
        "mock": NoopActuator,
        "airbot_real": AirbotActuator,
    }
    observer_cls = observer_cls.get(cfg.observer.name)
    if observer_cls is None:
        raise ValueError(f"Unsupported observer.name={cfg.observer.name!r}")
    if observer_cls is AirbotActuator:
        return observer_cls.from_config(cfg, robot=shared_robot)
    return observer_cls.from_config(cfg)

def build_executor(cfg: Config, shared_robot=None) -> BaseExecutor:
    executor_map = {
        "raw_action": RawActionExecutor,
        "ondevice_mpc": OnDeviceMpcExecutor,
    }
    executor_cls = executor_map.get(cfg.executor.name)
    if executor_cls is None:
        raise ValueError(f"Unsupported executor.name={cfg.executor.name!r}")
    actuator = _build_actuator(cfg, shared_robot=shared_robot)
    return executor_cls.from_config(cfg, actuator)

def build_runtime_components(cfg: Config, pending_actions_provider):
    shared_robot = _build_shared_airbot_robot(cfg)
    executor = build_executor(cfg, shared_robot=shared_robot)
    observer = build_observer(cfg, pending_actions_provider, shared_robot=shared_robot)
    return executor, observer
