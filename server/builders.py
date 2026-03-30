from __future__ import annotations

from config import Config
from model import BaseModelAdapter
from model import OpenPiRTCJaxAdapter
from model import OpenPiRTCTritonAdapter
from optimizer import BaseOptimizer
from optimizer import PassThroughOptimizer
from optimizer import TimeParameterizationMPC

def build_model(cfg: Config) -> BaseModelAdapter:
    adapter_map = {
        "openpi_rtc_jax": OpenPiRTCJaxAdapter,
        "openpi_rtc_triton": OpenPiRTCTritonAdapter,
    }
    adapter_cls = adapter_map.get(cfg.model.adapter)
    if adapter_cls is None:
        raise ValueError(f"Unsupported model.adapter={cfg.model.adapter!r}")
    return adapter_cls.from_config(cfg.model)

def build_optimizer(cfg: Config) -> BaseOptimizer:
    optimizer_map = {
        "pass_through": PassThroughOptimizer,
        "timeaxis_smooth": TimeParameterizationMPC,
    }
    optimizer_cls = optimizer_map.get(cfg.inference.optimizer)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported inference.optimizer={cfg.inference.optimizer!r}")
    return optimizer_cls.from_config(cfg.inference)
