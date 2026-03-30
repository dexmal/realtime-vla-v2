from __future__ import annotations
import argparse
import cv2
import logging
import pickle
import threading
import time
from pathlib import Path
import numpy as np
import requests
from builders import build_runtime_components
from config import load_config
from executor import OnDeviceMpcExecutor
from visualize import TrajectoryRecorder

logger = logging.getLogger(__name__)

class InferClient:
    def __init__(self, base_url: str, endpoint: str, timeout_s: float):
        self._session = requests.Session()
        self._url = f"{base_url.rstrip('/')}{endpoint}"
        self._timeout_s = float(timeout_s)

    def infer(self, payload: dict) -> dict:
        resp = self._session.post(
            self._url,
            data=pickle.dumps(payload),
            headers={"Content-Type": "application/octet-stream"},
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        result = pickle.loads(resp.content)
        if isinstance(result, dict):
            return result
        return {"action_list": result}

class MultiThreadedWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.executor, self.observer = build_runtime_components(
            cfg,
            pending_actions_provider=self._pending_count,
        )
        self.client = InferClient(base_url=cfg.client.infer_url, endpoint=cfg.client.endpoint, timeout_s=cfg.client.timeout_s)
        trajectory_family = "mpc" if isinstance(self.executor, OnDeviceMpcExecutor) else "smooth"
        self.recorder = TrajectoryRecorder(
            output_dir=cfg.visualization.output_dir,
            enabled=bool(cfg.visualization.enable_recording),
            record_videos=bool(cfg.visualization.record_videos),
            record_rerun=bool(cfg.visualization.record_rerun),
            trajectory_family=trajectory_family,
            max_pending_video_frames=int(cfg.visualization.max_pending_video_frames),
            video_fps=float(cfg.observer.fps),
        )

        self._collecting = False
        self._threads: list[threading.Thread] = []
        self._infer_latency_s = 0.2
        self._onlyinfer_s = 0.0
        self._default_run_duration_s = float(self.cfg.client.run_duration_s)
        self._is_mpc = isinstance(self.executor, OnDeviceMpcExecutor)
        self._has_control_thread = bool(self.executor.has_control_thread()) if not self._is_mpc else False
        self._actual_trace_interval_s = max(0.0, float(self.executor.get_control_dt_s()))
        self._actual_state_delay_s = max(0.0, float(self.cfg.executor.state_delay_s))
        self._last_actual_trace_ts = float("-inf")

        self._joint_state: list[float] | None = None
        self._joint_timestamp: float = 0.0
        self._joint_state_lock = threading.Lock()
        self._latest_infer_obs: dict | None = None
        self._infer_obs_lock = threading.Lock()

    def _pending_count(self) -> int:
        return int(self.executor.get_pending_count())

    def _recorder_needs_image_bytes(self) -> bool:
        return bool(self.recorder.enabled and (self.recorder.record_videos or self.recorder.record_rerun))

    def _fetch_image_observation(self) -> dict | None:
        raw_fetch = getattr(self.observer, "get_raw_image_observation", None)
        if callable(raw_fetch):
            raw_obs = raw_fetch()
            if raw_obs is not None:
                return raw_obs
        return self.observer.get_image_observation()

    @staticmethod
    def _has_images(observation: dict | None) -> bool:
        if not observation:
            return False
        return bool(observation.get("images") or observation.get("raw_images"))

    @staticmethod
    def _encode_images(images: dict | None) -> dict[str, bytes]:
        encoded: dict[str, bytes] = {}
        for camera_name, image_value in (images or {}).items():
            if image_value is None:
                continue
            if isinstance(image_value, (bytes, bytearray, memoryview)):
                encoded[str(camera_name)] = bytes(image_value)
                continue
            frame = np.asarray(image_value)
            if frame.size == 0:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError(f"Failed to encode image for camera {camera_name}")
            encoded[str(camera_name)] = buf.tobytes()
        return encoded

    def _materialize_images(self, observation: dict) -> dict[str, bytes]:
        images = observation.get("images") or {}
        if images:
            return self._encode_images(images)
        raw_images = observation.get("raw_images") or {}
        if not raw_images:
            return {}
        encoded = self._encode_images(raw_images)
        if encoded:
            observation["images"] = encoded
        return encoded

    def _record_image_observation(self, observation: dict) -> None:
        if not self._recorder_needs_image_bytes():
            return
        encoded_images = self._materialize_images(observation)
        if not encoded_images:
            return
        recorder_obs = dict(observation)
        recorder_obs["images"] = encoded_images
        recorder_obs.pop("raw_images", None)
        self.recorder.add_observation(recorder_obs)

    def _merge_image_observation(self, image_obs: dict, state: list[float] | None, state_ts: float) -> dict:
        image_ts = float(image_obs.get("image_timestamp", image_obs.get("timestamp", time.time())))
        merged_obs = {
            "state": state,
            "state_timestamp": state_ts if state is not None else None,
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": image_obs.get("pending_actions", self._pending_count()),
        }
        raw_images = image_obs.get("raw_images") or {}
        if raw_images:
            merged_obs["raw_images"] = raw_images
        images = image_obs.get("images") or {}
        if images:
            merged_obs["images"] = images
        return merged_obs

    def _record_aligned_actual_state(self, state: list[float], state_ts: float) -> None:
        if not self.recorder.enabled:
            return
        aligned_ts = float(state_ts) - self._actual_state_delay_s
        if aligned_ts <= self._last_actual_trace_ts:
            return
        if (
            self._actual_trace_interval_s > 0.0
            and aligned_ts - self._last_actual_trace_ts < self._actual_trace_interval_s
        ):
            return
        self._last_actual_trace_ts = aligned_ts
        self.recorder.add_record(
            {
                "timestamp": aligned_ts,
                "action": list(state),
                "source": "actual_action",
                "sync_mode": "measured_state_delay_shift",
                "observation_timestamp": float(state_ts),
                "state_delay_s": float(self._actual_state_delay_s),
            }
        )

    def _state_data_thread(self):
        while self._collecting:
            if self._is_mpc and self.executor.sleep_if_emergency():
                continue
            try:
                obs = self.observer.get_state_observation()
                if obs is None or obs.get("state") is None:
                    time.sleep(0.005)
                    continue
                state_ts = float(obs.get("state_timestamp", obs.get("timestamp", time.time())))
                state = list(obs["state"])
                with self._joint_state_lock:
                    self._joint_state = list(state)
                    self._joint_timestamp = state_ts
                if self._is_mpc:
                    self.executor.record_observation(state, state_ts)
                self._record_aligned_actual_state(state, state_ts)
                time.sleep(0.005)
            except Exception:
                time.sleep(0.005)

    def _image_data_thread(self):
        while self._collecting:
            if self._is_mpc and self.executor.sleep_if_emergency():
                continue
            try:
                image_obs = self._fetch_image_observation()
                if image_obs is None:
                    time.sleep(0.005)
                    continue
                state, state_ts = self._get_current_state()
                merged_obs = self._merge_image_observation(image_obs, state, state_ts)
                self._record_image_observation(merged_obs)
                with self._infer_obs_lock:
                    self._latest_infer_obs = dict(merged_obs)
            except Exception:
                time.sleep(0.005)

    def _get_current_state(self) -> tuple[list[float] | None, float]:
        with self._joint_state_lock:
            state = list(self._joint_state) if self._joint_state is not None else None
            ts = float(self._joint_timestamp)
        return state, ts

    def _get_latest_obs(self) -> dict | None:
        with self._infer_obs_lock:
            return dict(self._latest_infer_obs) if self._latest_infer_obs else None

    def _initialize_robot_pose(self) -> None:
        if not bool(self.cfg.executor.enable_init_action):
            return
        obs = self.observer.get_state_observation()
        current_state = None if obs is None else obs.get("state")
        if current_state is None:
            raise RuntimeError("Robot init failed: no state observation available")

        current = np.asarray(current_state, dtype=np.float32).reshape(-1)
        target = np.asarray(self.cfg.executor.init_action, dtype=np.float32).reshape(-1)
        if current.size == 0 or target.size == 0:
            raise RuntimeError("Robot init failed: empty init action or current state")
        if current.shape != target.shape:
            raise RuntimeError(
                f"Robot init failed: state shape {tuple(current.shape)} does not match init_action {tuple(target.shape)}"
            )

        steps = max(1, int(self.cfg.executor.init_steps))
        sleep_s = max(0.0, float(self.cfg.executor.init_sleep_s))
        for idx in range(steps):
            alpha = float(idx + 1) / float(steps)
            blended = current + (target - current) * alpha
            self.executor.apply_action(blended)
            if sleep_s > 0:
                time.sleep(sleep_s)
        with self._joint_state_lock:
            self._joint_state = None
            self._joint_timestamp = 0.0

    def _prime_startup_observation(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        latest_error: Exception | None = None
        image_obs: dict | None = None

        while time.monotonic() < deadline:
            try:
                image_candidate = self._fetch_image_observation()
            except Exception as exc:
                latest_error = exc
                time.sleep(0.05)
                continue
            if self._has_images(image_candidate):
                image_obs = image_candidate
                break
            time.sleep(0.01)

        if image_obs is None:
            detail = f": {latest_error}" if latest_error is not None else ""
            raise RuntimeError(f"Timed out waiting for the first camera frame after startup{detail}")

        state_obs = None
        while time.monotonic() < deadline:
            try:
                state_candidate = self.observer.get_state_observation()
            except Exception as exc:
                latest_error = exc
                time.sleep(0.01)
                continue
            if state_candidate and state_candidate.get("state") is not None:
                state_obs = state_candidate
                break
            time.sleep(0.01)

        if state_obs is None:
            detail = f": {latest_error}" if latest_error is not None else ""
            raise RuntimeError(f"Timed out waiting for the first robot state after startup{detail}")

        state = list(state_obs["state"])
        state_ts = float(state_obs.get("state_timestamp", state_obs.get("timestamp", time.time())))
        merged_obs = self._merge_image_observation(image_obs, list(state), state_ts)

        with self._joint_state_lock:
            self._joint_state = list(state)
            self._joint_timestamp = state_ts
        self._record_aligned_actual_state(state, state_ts)
        self._record_image_observation(merged_obs)
        with self._infer_obs_lock:
            self._latest_infer_obs = dict(merged_obs)

        if self._is_mpc and hasattr(self.executor, "prime_startup"):
            self.executor.prime_startup(
                anchor_action=state,
                anchor_timestamp=state_ts,
                bootstrap_action=True,
            )
            self.executor.record_observation(state, state_ts)
            logger.info(
                "Primed MPC startup with a measured anchor state at %.3f before enabling heartbeat.",
                state_ts,
            )

    def _heartbeat_thread(self):
        if self._is_mpc:
            dt_s = max(0.001, float(self.executor.get_control_dt_s()))
            next_tick = time.monotonic()
            while self._collecting:
                if self.executor.sleep_if_emergency():
                    next_tick = time.monotonic() + dt_s
                    continue
                now_mono = time.monotonic()
                if now_mono < next_tick:
                    time.sleep(next_tick - now_mono)
                elif now_mono - next_tick > dt_s:
                    next_tick = now_mono

                state, ts = self._get_current_state()
                if state is None:
                    next_tick += dt_s
                    continue
                result = self.executor.tick(state)
                if result is not None:
                    for record in result:
                        self.recorder.add_record(record)
                next_tick += dt_s
            return

        while self._collecting:
            try:
                loop_start = time.time()
                state, state_ts = self._get_current_state()
                interval_s, records = self.executor.heartbeat_step(state, state_ts)
                for record in records:
                    self.recorder.add_record(record)
                elapsed = time.time() - loop_start
                sleep_s = interval_s - elapsed
                time.sleep(sleep_s if sleep_s > 0 else 0)
            except Exception:
                time.sleep(0.01)

    def _control_thread(self):
        while self._collecting:
            try:
                loop_start = time.time()
                state, state_ts = self._get_current_state()
                interval_s, records = self.executor.control_step(state, state_ts)
                for record in records:
                    self.recorder.add_record(record)
                elapsed = time.time() - loop_start
                sleep_s = interval_s - elapsed
                time.sleep(sleep_s if sleep_s > 0 else 0)
            except Exception:
                time.sleep(0.01)

    def _inference_thread(self):
        while self._collecting:
            if self._is_mpc and self.executor.sleep_if_emergency():
                continue
            state, _ = self._get_current_state()
            obs = self._get_latest_obs()
            if state is None or obs is None or not self._has_images(obs):
                time.sleep(0.01)
                continue
            obs_image_ts = float(obs.get("image_timestamp", obs.get("timestamp", 0.0)) or 0.0)
            images = self._materialize_images(obs)
            if not images:
                time.sleep(0.01)
                continue

            if self._is_mpc:
                context, state_trajectory = self.executor.prepare_infer_context(
                    latency_s=self._infer_latency_s,
                    current_state=state,
                    image_timestamp=obs_image_ts,
                )
                if not state_trajectory:
                    time.sleep(0.01)
                    continue
                payload = {
                    "images": images,
                    "action": state_trajectory,
                    "state_delta": context.get("predicted_steps", context["predicted_idx"]),
                    "timestamp": float(context.get("request_time", time.time())),
                }
            else:
                payload, context = self.executor.prepare_infer_payload(
                    current_state=state,
                    images=images,
                    image_timestamp=obs_image_ts,
                    infer_latency_s=self._infer_latency_s,
                )
                if payload is None or context is None:
                    time.sleep(0.01)
                    continue

            send_time = time.time()
            try:
                resp = self.client.infer(payload)
            except Exception as exc:
                self.recorder.add_request_meta(
                    {
                        "timestamp": send_time,
                        "status": "error",
                        "error": str(exc),
                        "context": context,
                    }
                )
                if self._is_mpc:
                    self.executor.on_infer_failure(str(exc), current_state=state)
                time.sleep(0.01)
                continue
            recv_time = time.time()
            self._infer_latency_s = recv_time - send_time
            self._onlyinfer_s = float(resp.get("infer_time", 0.0)) if isinstance(resp, dict) else 0.0
            action_list = resp.get("action_list") if isinstance(resp, dict) else resp
            raw_action_list = resp.get("raw_action_list") if isinstance(resp, dict) else None
            if not action_list:
                if self._is_mpc:
                    self.executor.on_infer_failure("empty action_list", current_state=state)
                continue

            if self._is_mpc:
                try:
                    normalized = self.executor.normalize_infer_actions(action_list)
                except Exception as exc:
                    self.executor.on_infer_failure(
                        f"invalid action_list: {exc}", current_state=state
                    )
                    continue
                if not normalized:
                    continue
                raw_normalized = None
                if raw_action_list:
                    try:
                        raw_normalized = self.executor.normalize_infer_actions(raw_action_list)
                    except Exception:
                        raw_normalized = None
                self.executor.update_actions(normalized, context, raw_action_list=raw_normalized)
            else:
                raw_infer_actions = None
                if isinstance(resp, dict):
                    raw_infer_actions = resp.get("raw_action_list")
                if raw_infer_actions is None:
                    raw_infer_actions = action_list
                try:
                    normalized = self.executor.normalize_infer_actions(action_list)
                except Exception:
                    continue
                try:
                    raw_normalized = [list(action) for action in raw_infer_actions]
                except Exception:
                    raw_normalized = None
                self.executor.on_infer_actions(normalized, context, raw_action_list=raw_normalized)
            self.recorder.add_request_meta(
                {
                    "timestamp": send_time,
                    "status": "ok",
                    "recv_timestamp": recv_time,
                    "roundtrip_latency_s": self._infer_latency_s,
                    "server_infer_time_s": self._onlyinfer_s,
                    "action_count": len(normalized),
                    "context": context,
                }
            )
            self.recorder.add_inference_marker(timestamp=recv_time, action_count=len(normalized))
            print(
                f"Infer latency={self._infer_latency_s:.3f}s "
                f"queue_len={self._pending_count()} "
                f"onlyinfer_s={self._onlyinfer_s:.3f}s"
            )

    def run(self, duration_s: float | None = None):
        if duration_s is None:
            duration_s = self._default_run_duration_s
        self._initialize_robot_pose()
        if hasattr(self.observer, "start"):
            self.observer.start()
        if hasattr(self.observer, "drop_frame"):
            try:
                self.observer.drop_frame()
            except Exception:
                pass
        self._prime_startup_observation()
        self._collecting = True
        self._threads = [
            threading.Thread(target=self._state_data_thread, daemon=True),
            threading.Thread(target=self._image_data_thread, daemon=True),
            threading.Thread(target=self._heartbeat_thread, daemon=True),
            threading.Thread(target=self._inference_thread, daemon=True),
        ]
        if self._has_control_thread:
            self._threads.insert(
                3,
                threading.Thread(target=self._control_thread, daemon=True),
            )

        for thread in self._threads:
            thread.start()

        start = time.time()
        try:
            while time.time() - start < float(duration_s):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self._collecting = False
            for thread in self._threads:
                if thread.is_alive():
                    thread.join(timeout=2.0)
            self.observer.close()
            self.executor.close()

    def save_outputs(self, output_dir: str):
        if not bool(self.cfg.visualization.enable_recording):
            return {}
        self.recorder.flush()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    worker = MultiThreadedWorker(cfg)
    worker.run(duration_s=args.duration)
    worker.save_outputs(cfg.visualization.output_dir)

if __name__ == "__main__":
    main()
