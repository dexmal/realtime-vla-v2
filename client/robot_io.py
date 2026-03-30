from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from airbot_py.arm import AIRBOTPlay
    from airbot_py.arm import RobotMode
    from airbot_py.arm import SpeedProfile
except Exception:
    AIRBOTPlay = None
    RobotMode = None
    SpeedProfile = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


logger = logging.getLogger(__name__)


class BaseActuator(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    @abstractmethod
    def apply(self, action: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class BaseObserver(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, pending_actions_provider):
        raise NotImplementedError

    @abstractmethod
    def __post_init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state_observation(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_image_observation(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def get_raw_image_observation(self) -> dict | None:
        return None

    def drop_frame(self) -> None:
        return

    def start(self) -> None:
        return


@dataclass
class NoopActuator(BaseActuator):
    @classmethod
    def from_config(cls, cfg):
        return cls()

    def __post_init__(self) -> None:
        return

    def apply(self, action: np.ndarray) -> None:
        return

    def close(self) -> None:
        return


class _AirbotRobot:
    def __init__(self, airbot_host: str, left_port: int, right_port: int):
        if AIRBOTPlay is None or RobotMode is None or SpeedProfile is None:
            raise RuntimeError("airbot_py is not available")
        self._left_arm = AIRBOTPlay(url=airbot_host, port=int(left_port))
        self._right_arm = AIRBOTPlay(url=airbot_host, port=int(right_port))
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        self._left_arm.connect()
        self._right_arm.connect()
        self._left_arm.set_speed_profile(SpeedProfile.FAST)
        self._right_arm.set_speed_profile(SpeedProfile.FAST)
        self._left_arm.switch_mode(RobotMode.SERVO_JOINT_POS)
        self._right_arm.switch_mode(RobotMode.SERVO_JOINT_POS)
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._left_arm.disconnect()
        self._right_arm.disconnect()
        self._connected = False

    def get_joint_state(self) -> tuple[list[float], float]:
        timestamp = time.time()
        left_pos = self._left_arm.get_joint_pos()
        left_eef_pos = self._left_arm.get_eef_pos()
        right_pos = self._right_arm.get_joint_pos()
        right_eef_pos = self._right_arm.get_eef_pos()
        state = list(left_pos[:6]) + [float(np.asarray(left_eef_pos, dtype=np.float32).reshape(-1)[0])] + list(
            right_pos[:6]
        ) + [float(np.asarray(right_eef_pos, dtype=np.float32).reshape(-1)[0])]
        return state, timestamp

    def send_action(self, action: np.ndarray, left_gripper_bias: float = 0.0, right_gripper_bias: float = 0.0) -> None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size < 14:
            return
        self._left_arm.servo_joint_pos([float(v) for v in action_arr[:6]])
        self._left_arm.servo_eef_pos([float(action_arr[6]) + float(left_gripper_bias)])
        self._right_arm.servo_joint_pos([float(v) for v in action_arr[7:13]])
        self._right_arm.servo_eef_pos([float(action_arr[13]) + float(right_gripper_bias)])


def create_airbot_robot(airbot_host: str, left_port: int, right_port: int) -> _AirbotRobot:
    return _AirbotRobot(airbot_host=airbot_host, left_port=left_port, right_port=right_port)


@dataclass
class AirbotActuator(BaseActuator):
    airbot_host: str
    left_port: int
    right_port: int
    left_gripper_bias: float = 0.0
    right_gripper_bias: float = 0.0
    robot: _AirbotRobot | None = None

    @classmethod
    def from_config(cls, cfg, robot: _AirbotRobot | None = None):
        ex = cfg.executor
        return cls(
            airbot_host=ex.airbot_host,
            left_port=ex.airbot_left_port,
            right_port=ex.airbot_right_port,
            left_gripper_bias=ex.left_gripper_bias,
            right_gripper_bias=ex.right_gripper_bias,
            robot=robot,
        )

    def __post_init__(self) -> None:
        self._robot = self.robot or create_airbot_robot(
            self.airbot_host,
            self.left_port,
            self.right_port,
        )
        self._robot.connect()

    def apply(self, action: np.ndarray) -> None:
        self._robot.send_action(
            action,
            left_gripper_bias=self.left_gripper_bias,
            right_gripper_bias=self.right_gripper_bias,
        )

    def close(self) -> None:
        self._robot.disconnect()


def _encode_jpg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Failed to encode image as jpg")
    return buf.tobytes()


@dataclass
class MockStateObserver(BaseObserver):
    state_dim: int
    image_size: tuple[int, int]
    pending_actions_provider: callable = None

    @classmethod
    def from_config(cls, cfg, pending_actions_provider):
        obs = cfg.observer
        return cls(
            state_dim=obs.state_dim,
            image_size=obs.image_size,
            pending_actions_provider=pending_actions_provider,
        )

    def __post_init__(self) -> None:
        self._state_step = 0
        self._image_step = 0
        self._last_state: list[float] | None = None
        self._last_timestamp: float = 0.0

    def _generate_state(self) -> tuple[list[float], float]:
        self._state_step += 1
        now = time.time()
        phase = self._state_step * 0.03
        state = (0.5 * np.sin(np.arange(self.state_dim, dtype=np.float32) * 0.2 + phase)).tolist()
        self._last_state = state
        self._last_timestamp = now
        return state, now

    def _generate_raw_images(self) -> tuple[dict[str, np.ndarray], float]:
        self._image_step += 1
        now = time.time()
        h, w = self.image_size[1], self.image_size[0]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"frame={self._image_step}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"state={self._state_step}",
            (24, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 180, 255),
            2,
        )
        return {
            "high": frame.copy(),
            "left_hand": frame.copy(),
            "right_hand": frame.copy(),
        }, now

    def get_state_observation(self) -> dict:
        state, state_ts = self._generate_state()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "state": state,
            "action": state,
            "state_timestamp": state_ts,
            "timestamp": state_ts,
            "pending_actions": pending,
        }

    def get_image_observation(self) -> dict:
        raw_images, image_ts = self._generate_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "images": {name: _encode_jpg(frame) for name, frame in raw_images.items()},
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def get_raw_image_observation(self) -> dict:
        raw_images, image_ts = self._generate_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "raw_images": raw_images,
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def close(self) -> None:
        return


class _RealSenseCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not available")
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.ready = False
        self.running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self.ready = True

    @staticmethod
    def _enable_global_time(profile) -> None:
        device = profile.get_device()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)

    def start(self) -> None:
        if not self.ready:
            raise RuntimeError(f"Camera {self.serial} is not ready")
        if self._started:
            return
        profile = self.pipeline.start(self.config)
        self._enable_global_time(profile)
        self.running = True

        def _update_frame() -> None:
            while self.running:
                try:
                    frame = self.pipeline.wait_for_frames(timeout_ms=2000)
                except Exception as exc:
                    logger.warning("Camera %s frame update failed: %s", self.serial, exc)
                    continue
                with self._lock:
                    self._frame = frame

        self._thread = threading.Thread(target=_update_frame, daemon=True)
        self._thread.start()
        self._started = True

    def get_frame(self) -> dict | None:
        if not self.ready:
            return None
        while self.running:
            with self._lock:
                frame = self._frame
                if frame is not None:
                    self._frame = None
                    break
            time.sleep(0.01)
        else:
            return None

        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()
        if color_frame is None:
            raise RuntimeError(f"Camera {self.serial}: color frame is None")
        color_img = np.asanyarray(color_frame.get_data())
        color_timestamp = float(color_frame.get_timestamp() / 1000.0)
        depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame is not None else None
        depth_timestamp = float(depth_frame.get_timestamp() / 1000.0) if depth_frame is not None else color_timestamp
        return {
            "color_image": color_img,
            "color_timestamp": color_timestamp,
            "depth_image": depth_img,
            "depth_timestamp": depth_timestamp,
        }

    def drop_frame(self) -> None:
        if not self.ready or not self._started:
            return
        with self._lock:
            self._frame = None

    def close(self) -> None:
        if not self.ready or not self._started:
            return
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.pipeline.stop()
        self._started = False


class _AirbotCameraRig:
    def __init__(self, top_camera_id: str, left_camera_id: str, right_camera_id: str, width: int, height: int, fps: int):
        self.left_camera = _RealSenseCamera(left_camera_id, width=width, height=height, fps=fps)
        self.right_camera = _RealSenseCamera(right_camera_id, width=width, height=height, fps=fps)
        self.top_camera = _RealSenseCamera(top_camera_id, width=width, height=height, fps=fps)

    def start(self) -> None:
        self.left_camera.start()
        self.right_camera.start()
        self.top_camera.start()

    def stop(self) -> None:
        self.left_camera.close()
        self.right_camera.close()
        self.top_camera.close()

    def get_frame(self) -> dict[str, dict | None]:
        return {
            "left_camera": self.left_camera.get_frame(),
            "right_camera": self.right_camera.get_frame(),
            "top_camera": self.top_camera.get_frame(),
        }

    def drop_frame(self) -> None:
        self.left_camera.drop_frame()
        self.right_camera.drop_frame()
        self.top_camera.drop_frame()


@dataclass
class AirbotRealObserver(BaseObserver):
    airbot_host: str
    left_port: int
    right_port: int
    top_camera_id: str
    left_camera_id: str
    right_camera_id: str
    enable_cameras: bool
    image_size: tuple[int, int]
    fps: int
    pending_actions_provider: callable = None
    robot: _AirbotRobot | None = None

    @classmethod
    def from_config(cls, cfg, pending_actions_provider, robot: _AirbotRobot | None = None):
        obs = cfg.observer
        return cls(
            airbot_host=obs.airbot_host,
            left_port=obs.airbot_left_port,
            right_port=obs.airbot_right_port,
            top_camera_id=obs.top_camera_id,
            left_camera_id=obs.left_camera_id,
            right_camera_id=obs.right_camera_id,
            enable_cameras=obs.enable_cameras,
            image_size=obs.image_size,
            fps=obs.fps,
            pending_actions_provider=pending_actions_provider,
            robot=robot,
        )

    def __post_init__(self) -> None:
        self._robot = self.robot or create_airbot_robot(
            self.airbot_host,
            self.left_port,
            self.right_port,
        )
        self._robot.connect()
        self._camera_rig: _AirbotCameraRig | None = None
        self._cameras_started = False
        if self.enable_cameras:
            w, h = int(self.image_size[0]), int(self.image_size[1])
            self._camera_rig = _AirbotCameraRig(
                top_camera_id=self.top_camera_id,
                left_camera_id=self.left_camera_id,
                right_camera_id=self.right_camera_id,
                width=w,
                height=h,
                fps=int(self.fps),
            )

    def _read_robot_state(self) -> tuple[list[float], float]:
        return self._robot.get_joint_state()

    def _resize_like_legacy(self, image: np.ndarray) -> np.ndarray:
        target_size = (int(self.image_size[0]), int(self.image_size[1]))
        if image.shape[1] == target_size[0] and image.shape[0] == target_size[1]:
            return image
        return cv2.resize(image, target_size)

    def _read_raw_images(self) -> tuple[dict[str, np.ndarray], float]:
        if not self.enable_cameras or self._camera_rig is None:
            return {}, time.time()
        frames = self._camera_rig.get_frame()
        left = frames.get("left_camera")
        right = frames.get("right_camera")
        top = frames.get("top_camera")
        if left is None or right is None or top is None:
            return {}, time.time()
        top_ts = top.get("color_timestamp")
        if top_ts is None:
            top_ts = time.time()
        return {
            "high": self._resize_like_legacy(top["color_image"]),
            "left_hand": self._resize_like_legacy(left["color_image"]),
            "right_hand": self._resize_like_legacy(right["color_image"]),
        }, float(top_ts)

    @staticmethod
    def _encode_images(raw_images: dict[str, np.ndarray]) -> dict[str, bytes]:
        return {name: _encode_jpg(frame) for name, frame in raw_images.items() if frame is not None}

    def get_state_observation(self) -> dict:
        state, state_ts = self._read_robot_state()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "state": state,
            "action": state,
            "state_timestamp": state_ts,
            "timestamp": state_ts,
            "pending_actions": pending,
        }

    def get_image_observation(self) -> dict:
        raw_images, image_ts = self._read_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "images": self._encode_images(raw_images),
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def get_raw_image_observation(self) -> dict:
        raw_images, image_ts = self._read_raw_images()
        pending = int(self.pending_actions_provider()) if self.pending_actions_provider is not None else 0
        return {
            "raw_images": raw_images,
            "image_timestamp": image_ts,
            "timestamp": image_ts,
            "pending_actions": pending,
        }

    def close(self) -> None:
        if self._camera_rig is not None:
            self._camera_rig.stop()
        self._cameras_started = False
        self._robot.disconnect()

    def drop_frame(self) -> None:
        if self._camera_rig is not None and self._cameras_started:
            self._camera_rig.drop_frame()

    def start(self) -> None:
        if self._camera_rig is not None and not self._cameras_started:
            self._camera_rig.start()
            self._cameras_started = True
