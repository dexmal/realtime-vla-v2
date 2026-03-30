from __future__ import annotations
from abc import ABC
from abc import abstractmethod
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

class BaseModelAdapter(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, model_cfg):
        raise NotImplementedError

    @abstractmethod
    def __post_init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _warmup_policy(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer_actions(self, state: dict) -> list:
        raise NotImplementedError

def _extract_state_sequence(state: dict) -> np.ndarray:
    raw_state_sequence = state.get("action") if "action" in state else state.get("actions", state.get("state"))
    if raw_state_sequence is None:
        raise ValueError("No state sequence found in request payload.")
    state_arr = np.asarray(raw_state_sequence, dtype=np.float32)
    if state_arr.ndim == 1:
        state_arr = state_arr[None, :]
    return state_arr

def _process_rtc_actions_for_robot(actions) -> np.ndarray:
    actions_array = np.asarray(actions, dtype=np.float32)
    reorder_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    reordered_actions = actions_array[..., reorder_indices]
    return reordered_actions

def _load_norm_stats(norm_stats_dir: str) -> dict[str, dict[str, np.ndarray]]:
    path = Path(norm_stats_dir)
    if path.is_dir():
        path = path / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"norm_stats.json not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        norm_stats = json.load(fp)["norm_stats"]
    return {
        "state": {
            "q01": np.asarray(norm_stats["state"]["q01"], dtype=np.float32),
            "q99": np.asarray(norm_stats["state"]["q99"], dtype=np.float32),
        },
        "actions": {
            "q01": np.asarray(norm_stats["actions"]["q01"], dtype=np.float32),
            "q99": np.asarray(norm_stats["actions"]["q99"], dtype=np.float32),
        },
    }

@dataclass
class OpenPiRTCJaxAdapter(BaseModelAdapter):
    config_name: str
    checkpoint_dir: str
    prompt: str
    adarms_knob: int
    valid_action_num: int
    action_type: str
    image_size: tuple[int, int]

    @classmethod
    def from_config(cls, model_cfg):
        return cls(
            config_name=model_cfg.config_name,
            checkpoint_dir=model_cfg.checkpoint,
            prompt=model_cfg.prompt,
            adarms_knob=model_cfg.adarms_knob,
            valid_action_num=model_cfg.valid_action_num,
            action_type=model_cfg.action_type,
            image_size=model_cfg.image_size,
        )

    def __post_init__(self) -> None:
        from openpi import transforms as _transforms
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        config = _config.get_config(self.config_name)
        self._policy = _policy_config.create_trained_policy(
            config,
            self.checkpoint_dir,
            repack_transforms=_transforms.Group(),
        )
        self._action_horizon = getattr(self._policy._model, "action_horizon", None)
        if self._action_horizon is None and hasattr(self._policy._model, "config"):
            self._action_horizon = self._policy._model.config.action_horizon
        if self._action_horizon is not None:
            self._action_horizon = int(self._action_horizon)
        self._adarms_knob = np.int32(self.adarms_knob)
        self._delta_mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1), dtype=bool)
        self._output_has_absolute = self._check_absolute_output(_transforms)
        self._key_mapping = {
            "high": "observation/cam_high",
            "left_hand": "observation/cam_wrist_left",
            "right_hand": "observation/cam_wrist_right",
        }
        self.inp_images = {
            key: np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            for key in self._key_mapping.values()
        }
        self._zero_state = np.zeros(14, dtype=np.float32)
        self._warmup_policy()

    def _warmup_policy(self) -> None:
        if self._action_horizon is None:
            max_prefill_len = 1
        else:
            max_prefill_len = self._action_horizon
        if max_prefill_len <= 0:
            return
        print(f"Warmup: compiling prefill lengths 1..{max_prefill_len}")
        warmup_actions = np.zeros((max_prefill_len, self._zero_state.shape[0]), dtype=np.float32)
        warmup_actions = self._pad_prefill_actions(warmup_actions)
        inputs = {
            **self.inp_images,
            "state": self._zero_state,
            "prompt": self.prompt,
            "adarms_knob": self._adarms_knob,
            "actions": warmup_actions,
        }
        saved_rng = getattr(self._policy, "_rng", None)
        start_time = time.time()
        for prefill_len in range(1, max_prefill_len + 1):
            inputs["action_prefill_len"] = np.int32(prefill_len)
            self._policy.infer(inputs)
        if saved_rng is not None:
            self._policy._rng = saved_rng
        print(f"Warmup complete in {time.time() - start_time:.2f}s")

    def _check_absolute_output(self, _transforms) -> bool:
        output_transform = getattr(self._policy, "_output_transform", None)
        transforms_list = getattr(output_transform, "transforms", None)
        if transforms_list is None:
            return False
        return any(isinstance(t, _transforms.AbsoluteActions) for t in transforms_list)

    def process_images(self, state: dict, image_type: list[str]) -> None:
        for source in image_type:
            if source in state.get("images", {}):
                image_data = state["images"][source]
                image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                final_key = self._key_mapping.get(source, source)
                self.inp_images[final_key] = image

    def process_actions_for_model(self, actions) -> np.ndarray:
        return actions

    def process_state_sequence_for_model(self, state_sequence) -> np.ndarray:
        state_sequence = np.asarray(state_sequence, dtype=np.float32)
        if state_sequence.ndim == 1:
            return self.process_actions_for_model(state_sequence)
        return np.stack([self.process_actions_for_model(s) for s in state_sequence], axis=0)

    def _pad_prefill_actions(self, prefill_actions: np.ndarray) -> np.ndarray:
        prefill_actions = np.asarray(prefill_actions, dtype=np.float32)
        if prefill_actions.ndim == 1:
            prefill_actions = prefill_actions[None, :]
        if self._action_horizon is None:
            return prefill_actions
        target = int(self._action_horizon)
        if prefill_actions.shape[0] < target:
            pad = np.zeros((target - prefill_actions.shape[0], prefill_actions.shape[1]), dtype=prefill_actions.dtype)
            return np.concatenate([prefill_actions, pad], axis=0)
        return prefill_actions[:target]

    def _to_absolute_actions(self, actions: np.ndarray, obs_state: np.ndarray) -> np.ndarray:
        dims = min(actions.shape[-1], obs_state.shape[-1], self._delta_mask.shape[0])
        if dims <= 0:
            return actions
        actions = actions.copy()
        actions[..., :dims] += obs_state[:dims] * self._delta_mask[:dims]
        return actions

    def process_actions_for_robot(self, actions) -> np.ndarray:
        return _process_rtc_actions_for_robot(actions)

    def infer_actions(self, state: dict) -> list:
        image_type = list(self._key_mapping.keys())
        self.process_images(state, image_type)
        state_sequence = self.process_state_sequence_for_model(_extract_state_sequence(state))
        if state_sequence.ndim == 1:
            state_sequence = state_sequence[None, :]
        obs_state = state_sequence[0]
        prefill_len = state_sequence.shape[0]
        if self._action_horizon is not None:
            prefill_len = min(prefill_len, int(self._action_horizon))
        prefill_actions = self._pad_prefill_actions(state_sequence[:prefill_len])

        inputs = {
            **self.inp_images,
            "state": obs_state,
            "prompt": self.prompt,
            "adarms_knob": self._adarms_knob,
        }
        if prefill_len > 0:
            inputs["actions"] = prefill_actions
            inputs["action_prefill_len"] = np.int32(prefill_len)
        result = self._policy.infer(inputs)["actions"]
        if not self._output_has_absolute:
            result = self._to_absolute_actions(result, obs_state)
        result = result[min(prefill_len, result.shape[0]):]
        if self.valid_action_num is not None:
            result = result[: int(self.valid_action_num)]
        return self.process_actions_for_robot(result).tolist()

@dataclass
class OpenPiRTCTritonAdapter(BaseModelAdapter):
    checkpoint_dir: str
    prompt: str
    adarms_knob: int
    valid_action_num: int
    action_type: str
    image_size: tuple[int, int]
    action_horizon: int
    tokenizer_path: str
    norm_stats_dir: str
    discrete_state_input: bool = True
    state_dim: int = 14
    action_dim: int = 14
    noise_seed: int | None = None

    @classmethod
    def from_config(cls, model_cfg):
        return cls(
            checkpoint_dir=model_cfg.checkpoint,
            prompt=model_cfg.prompt,
            adarms_knob=model_cfg.adarms_knob,
            valid_action_num=model_cfg.valid_action_num,
            action_type=model_cfg.action_type,
            image_size=model_cfg.image_size,
            action_horizon=model_cfg.action_horizon,
            tokenizer_path=model_cfg.tokenizer_path,
            norm_stats_dir=model_cfg.norm_stats_dir,
            discrete_state_input=model_cfg.discrete_state_input,
            state_dim=model_cfg.state_dim,
            action_dim=model_cfg.action_dim,
            noise_seed=model_cfg.noise_seed,
        )

    def __post_init__(self) -> None:
        import torch
        from pi05rtc_infer import Pi05RTCInference

        if self.discrete_state_input and not self.tokenizer_path:
            raise ValueError("tokenizer_path is required for RTC Triton when discrete_state_input=True.")
        if not self.norm_stats_dir:
            raise ValueError("norm_stats_dir is required for RTC Triton inference.")

        with open(self.checkpoint_dir, "rb") as fp:
            weights = pickle.load(fp)
        self._policy = Pi05RTCInference(
            checkpoint=weights,
            num_views=3,
            chunk_size=int(self.action_horizon),
            tokenizer_path=self.tokenizer_path or None,
            max_tokenize_len=200,
            max_prompt_text=self.prompt,
            discrete_state_input=bool(self.discrete_state_input),
            state_dim_for_max_prompt=int(self.state_dim),
        )
        self._torch = torch
        self._adarms_knob = np.int32(self.adarms_knob)
        self._digitize_bins = np.linspace(-1, 1, 256 + 1)[:-1]
        self._norm_stats = _load_norm_stats(self.norm_stats_dir)
        self._state_q01 = self._norm_stats["state"]["q01"]
        self._state_q99 = self._norm_stats["state"]["q99"]
        self._actions_q01 = self._norm_stats["actions"]["q01"]
        self._actions_q99 = self._norm_stats["actions"]["q99"]
        self._key_mapping = {
            "high": "observation/cam_high",
            "left_hand": "observation/cam_wrist_left",
            "right_hand": "observation/cam_wrist_right",
        }
        self.inp_images = {
            key: np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            for key in self._key_mapping.values()
        }
        self._observation_images_cpu = self._torch.empty(
            (3, 224, 224, 3),
            dtype=self._torch.float32,
            pin_memory=True,
        )
        self._observation_images_cpu_np = self._observation_images_cpu.numpy()
        self._observation_images_gpu = self._torch.empty(
            (3, 224, 224, 3),
            dtype=self._torch.bfloat16,
            device="cuda",
        )
        self._diffusion_noise_cpu = self._torch.empty(
            (int(self.action_horizon), 32),
            dtype=self._torch.float32,
            pin_memory=True,
        )
        self._diffusion_noise_cpu_np = self._diffusion_noise_cpu.numpy()
        self._diffusion_noise_gpu = self._torch.empty(
            (int(self.action_horizon), 32),
            dtype=self._torch.bfloat16,
            device="cuda",
        )
        self._rng = np.random.default_rng(self.noise_seed)
        self._warmup_policy()

    def _warmup_policy(self) -> None:
        max_prefill_len = int(self.action_horizon)
        if max_prefill_len <= 0:
            return

        print(f"Warmup: compiling prefill lengths 1..{max_prefill_len}")
        zero_state = np.zeros((int(self.state_dim),), dtype=np.float32)
        state_norm = self._normalize_state(zero_state, target_dim=32)
        state_tokens = self._digitize_state(state_norm[: self.state_dim])
        prefill_actions = np.zeros((max_prefill_len, 32), dtype=np.float32)
        start_time = time.time()

        for prefill_len in range(1, max_prefill_len + 1):
            self._observation_images_gpu.zero_()
            self._diffusion_noise_gpu.zero_()
            self._policy.forward(
                self._observation_images_gpu,
                self._diffusion_noise_gpu,
                task_prompt=self.prompt,
                state_tokens=state_tokens,
                action_prefill_len=prefill_len,
                prefill_actions=prefill_actions,
            )

        print(f"Warmup complete in {time.time() - start_time:.2f}s")

    def process_images(self, state: dict, image_type: list[str]) -> None:
        for source in image_type:
            if source not in state.get("images", {}):
                continue
            image_data = state["images"][source]
            image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            final_key = self._key_mapping.get(source, source)
            self.inp_images[final_key] = image

    def process_actions_for_model(self, actions) -> np.ndarray:
        return np.asarray(actions, dtype=np.float32)

    def process_state_sequence_for_model(self, state_sequence) -> np.ndarray:
        state_sequence = np.asarray(state_sequence, dtype=np.float32)
        if state_sequence.ndim == 1:
            return self.process_actions_for_model(state_sequence)
        return np.stack([self.process_actions_for_model(s) for s in state_sequence], axis=0)

    def _resize_with_pad(self, image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
        from PIL import Image

        pil_image = Image.fromarray(image)
        cur_width, cur_height = pil_image.size
        if cur_width == width and cur_height == height:
            return image
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        zero_image = Image.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        return np.array(zero_image)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0 * 2.0 - 1.0

    def _pad_to_dim(self, x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
        cur = x.shape[axis]
        if cur >= target_dim:
            return x
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (0, target_dim - cur)
        return np.pad(x, pad_width, constant_values=value)

    def _normalize_state(self, state: np.ndarray, target_dim: int = 32) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        norm_dim = min(
            int(self.state_dim),
            state.shape[-1],
            self._state_q01.shape[-1],
            self._state_q99.shape[-1],
        )
        state_norm = (state[:norm_dim] - self._state_q01[:norm_dim]) / (
            self._state_q99[:norm_dim] - self._state_q01[:norm_dim] + 1e-6
        ) * 2.0 - 1.0
        return self._pad_to_dim(state_norm.astype(np.float32), target_dim, axis=-1, value=0.0)

    def _digitize_state(self, state_normed: np.ndarray) -> np.ndarray:
        return (np.digitize(state_normed, bins=self._digitize_bins) - 1).astype(np.int32)

    def _delta_mask(self, dim: int) -> np.ndarray:
        mask = np.ones((dim,), dtype=bool)
        for i in range(6, dim, 7):
            mask[i] = False
        return mask

    def _normalize_prefill_actions(
        self,
        raw_actions: np.ndarray,
        raw_state: np.ndarray,
        target_dim: int = 32,
    ) -> np.ndarray:
        raw_actions = np.asarray(raw_actions, dtype=np.float32)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        if raw_actions.ndim != 2:
            raise ValueError(f"prefill actions must be [T,D], got {raw_actions.shape}")
        if raw_actions.shape[-1] < self.action_dim:
            raise ValueError(f"prefill action dim must be >= {self.action_dim}, got {raw_actions.shape[-1]}")

        acts = raw_actions[:, : self.action_dim].copy()
        mask = self._delta_mask(int(self.action_dim))
        acts -= np.where(mask, raw_state[: self.action_dim], 0.0)[None, :]
        acts = (acts - self._actions_q01[: self.action_dim]) / (
            self._actions_q99[: self.action_dim] - self._actions_q01[: self.action_dim] + 1e-6
        ) * 2.0 - 1.0
        return self._pad_to_dim(acts.astype(np.float32), target_dim, axis=-1, value=0.0)

    def _unnormalize_actions(self, actions: np.ndarray, target_dim: int = 32) -> np.ndarray:
        q01 = self._pad_to_dim(self._actions_q01, target_dim)
        q99 = self._pad_to_dim(self._actions_q99, target_dim)
        return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def _to_absolute_actions(self, delta_actions: np.ndarray, raw_state: np.ndarray) -> np.ndarray:
        actions = np.asarray(delta_actions, dtype=np.float32).copy()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        actions[..., : self.action_dim] += raw_state[: self.action_dim][None, :]
        for i in range(self.action_dim // 7):
            actions[..., (i + 1) * 7 - 1] -= raw_state[(i + 1) * 7 - 1]
        return actions

    def _build_observation_images(self):
        for idx, key in enumerate((
            "observation/cam_high",
            "observation/cam_wrist_left",
            "observation/cam_wrist_right",
        )):
            img = self._resize_with_pad(self.inp_images[key], 224, 224)
            img = self._normalize_image(img)
            np.copyto(self._observation_images_cpu_np[idx], img, casting="no")
        self._observation_images_gpu.copy_(self._observation_images_cpu, non_blocking=True)
        return self._observation_images_gpu

    def _build_diffusion_noise(self):
        noise = self._rng.standard_normal((int(self.action_horizon), 32)).astype(np.float32)
        np.copyto(self._diffusion_noise_cpu_np, noise, casting="no")
        self._diffusion_noise_gpu.copy_(self._diffusion_noise_cpu, non_blocking=True)
        return self._diffusion_noise_gpu

    def process_actions_for_robot(self, actions) -> np.ndarray:
        return _process_rtc_actions_for_robot(actions)

    def infer_actions(self, state: dict) -> list:
        image_type = list(self._key_mapping.keys())
        self.process_images(state, image_type)
        state_sequence = self.process_state_sequence_for_model(_extract_state_sequence(state))
        if state_sequence.ndim == 1:
            state_sequence = state_sequence[None, :]
        obs_state = state_sequence[0]
        prefill_len = min(state_sequence.shape[0], int(self.action_horizon))
        prefill_actions_raw = state_sequence[:prefill_len] if prefill_len > 0 else None
        state_norm = self._normalize_state(obs_state, target_dim=32)
        state_tokens = self._digitize_state(state_norm[: self.state_dim])
        observation_images = self._build_observation_images()
        diffusion_noise = self._build_diffusion_noise()

        prefill_actions = None
        if prefill_len > 0:
            prefill_actions = self._normalize_prefill_actions(prefill_actions_raw, obs_state, target_dim=32)

        result = self._policy.forward(
            observation_images,
            diffusion_noise,
            task_prompt=self.prompt,
            state_tokens=state_tokens,
            action_prefill_len=prefill_len if prefill_len > 0 else None,
            prefill_actions=prefill_actions,
        )
        result = result.detach().cpu().float().numpy()
        result = self._unnormalize_actions(result, target_dim=32)[:, : self.action_dim]
        result = self._to_absolute_actions(result, obs_state)
        result = result[min(prefill_len, result.shape[0]) :]
        if self.valid_action_num is not None:
            result = result[: int(self.valid_action_num)]
        return self.process_actions_for_robot(result).tolist()
