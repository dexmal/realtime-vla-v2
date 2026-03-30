from __future__ import annotations
import json
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except Exception:
    rr = None
    rrb = None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
class _BaseVideoSink:
    backend_name: str
    output_path: Path

    def write_encoded_frame(self, image_bytes: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class _OpenCvMp4Sink(_BaseVideoSink):
    backend_name = "opencv_mp4"

    def __init__(self, output_path: Path, fps: float) -> None:
        self.output_path = output_path
        self._fps = max(1.0, float(fps))
        self._writer: cv2.VideoWriter | None = None

    def write_encoded_frame(self, image_bytes: bytes) -> None:
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return
        if self._writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to create video writer for {self.output_path}")
            self._writer = writer
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class _FfmpegMjpegSink(_BaseVideoSink):
    backend_name = "ffmpeg_mjpeg_copy"

    def __init__(self, ffmpeg_path: str, output_path: Path, fps: float) -> None:
        self.output_path = output_path
        self._process = subprocess.Popen(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "image2pipe",
                "-framerate",
                f"{max(1.0, float(fps)):0.6f}",
                "-vcodec",
                "mjpeg",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "copy",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_encoded_frame(self, image_bytes: bytes) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"ffmpeg stdin is not available for {self.output_path}")
        try:
            self._process.stdin.write(image_bytes)
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg pipe closed while writing {self.output_path}") from exc

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        try:
            code = self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            code = self._process.wait(timeout=1.0)
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code} for {self.output_path}")


@dataclass
class TrajectoryRecorder:
    output_dir: str
    enabled: bool = True
    record_videos: bool = False
    record_rerun: bool = False
    trajectory_family: str = "mpc"
    max_pending_video_frames: int = 32
    video_fps: float = 30.0
    _writer_cond: threading.Condition = field(init=False, repr=False)
    _pending_jsonl_writes: deque[tuple[Path, dict[str, Any]]] = field(default_factory=deque, init=False, repr=False)
    _pending_video_writes: deque[tuple[str, float, bytes]] = field(default_factory=deque, init=False, repr=False)
    _pending_rerun_records: deque[dict[str, Any]] = field(default_factory=deque, init=False, repr=False)
    _pending_rerun_images: deque[tuple[float, dict[str, bytes]]] = field(default_factory=deque, init=False, repr=False)
    _writer_stop: bool = field(default=False, init=False, repr=False)
    _writer_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _video_sinks: dict[str, _BaseVideoSink] = field(default_factory=dict, init=False, repr=False)
    _rrd_dir_name: str = field(default="rrd", init=False, repr=False)
    _rerun_enabled: bool = field(default=False, init=False, repr=False)
    _rerun_path: str | None = field(default=None, init=False)
    _dropped_video_frames: int = field(default=0, init=False, repr=False)
    _dropped_rerun_frames: int = field(default=0, init=False, repr=False)
    _ffmpeg_path: str | None = field(default=None, init=False, repr=False)
    _video_backend: str = field(default="disabled", init=False, repr=False)
    _writer_error: Exception | None = field(default=None, init=False, repr=False)
    _rerun_series_paths: set[str] = field(default_factory=set, init=False, repr=False)
    _rerun_point_series_paths: set[str] = field(default_factory=set, init=False, repr=False)
    _latest_trajectory_records: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _pre_mpc_history: deque[dict[str, Any]] = field(default_factory=deque, init=False, repr=False)
    _pending_pre_mpc_markers: deque[dict[str, Any]] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.trajectory_family not in {"mpc", "smooth"}:
            raise ValueError(f"Unsupported trajectory_family={self.trajectory_family!r}")
        self._writer_cond = threading.Condition()
        rrd_dir = self._rrd_dir()
        if rrd_dir.exists():
            shutil.rmtree(rrd_dir)
        rrd_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_output_files()
        self._writer_thread = threading.Thread(target=self._jsonl_writer_thread, daemon=True)
        self._writer_thread.start()
        if self.record_videos:
            self._ffmpeg_path = shutil.which("ffmpeg")
            self._video_backend = (
                _FfmpegMjpegSink.backend_name if self._ffmpeg_path is not None else _OpenCvMp4Sink.backend_name
            )
        self._rerun_path = str(rrd_dir / "runtime.rrd")
        if rr is None or not self.record_rerun:
            return
        try:
            rr.init("openpi_client_runtime", spawn=False)
            rr.save(self._rerun_path)
            self._rerun_enabled = True
            self._apply_default_blueprint()
        except Exception:
            self._rerun_enabled = False

    def add_record(self, record: dict) -> None:
        if not self.enabled:
            return
        normalized = _to_jsonable(record)
        source = str(normalized.get("source", ""))
        marker_records: list[dict[str, Any]] = []
        if source in {
            "raw_pre_mpc_action",
            "pre_mpc_action",
            "post_mpc_action",
            "raw_pre_smooth_action",
            "pre_smooth_action",
            "post_smooth_action",
            "actual_action",
        }:
            marker_records = self._remember_trajectory_record(source, normalized)
        if source == "raw_pre_mpc_action":
            self._enqueue_jsonl(self._raw_pre_mpc_action_records_path(), normalized)
            self._enqueue_jsonl(self._mpc_action_records_path(), normalized)
        elif source == "pre_mpc_action":
            self._enqueue_jsonl(self._pre_mpc_action_records_path(), normalized)
            self._enqueue_jsonl(self._mpc_action_records_path(), normalized)
        elif source == "post_mpc_action":
            self._enqueue_jsonl(self._post_mpc_action_records_path(), normalized)
            self._enqueue_jsonl(self._mpc_action_records_path(), normalized)
        elif source == "raw_pre_smooth_action":
            self._enqueue_jsonl(self._raw_pre_smooth_action_records_path(), normalized)
            self._enqueue_jsonl(self._smooth_action_records_path(), normalized)
        elif source == "pre_smooth_action":
            self._enqueue_jsonl(self._pre_smooth_action_records_path(), normalized)
            self._enqueue_jsonl(self._smooth_action_records_path(), normalized)
        elif source == "post_smooth_action":
            self._enqueue_jsonl(self._post_smooth_action_records_path(), normalized)
            self._enqueue_jsonl(self._smooth_action_records_path(), normalized)
        elif source == "actual_action":
            self._enqueue_jsonl(self._actual_action_records_path(), normalized)
            if self.trajectory_family == "smooth":
                self._enqueue_jsonl(self._smooth_action_records_path(), normalized)
            else:
                self._enqueue_jsonl(self._mpc_action_records_path(), normalized)
        if self._rerun_enabled:
            self._enqueue_rerun_record(normalized)
            for marker_record in marker_records:
                self._enqueue_rerun_record(marker_record)

    def add_request_meta(self, meta: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._enqueue_jsonl(self._request_meta_records_path(), _to_jsonable(meta))

    def add_inference_marker(self, timestamp: float, action_count: int | None = None) -> None:
        if not self.enabled:
            return
        event = {"timestamp": float(timestamp)}
        if action_count is not None:
            event["action_count"] = int(action_count)
        self._enqueue_jsonl(self._inference_event_records_path(), event)
        if not self._rerun_enabled:
            return
        pending_marker = {"timestamp": float(timestamp)}
        if action_count is not None:
            pending_marker["action_count"] = int(action_count)
        with self._writer_cond:
            if self._writer_error is not None:
                return
            self._pending_pre_mpc_markers.append(pending_marker)
            for marker_record in self._drain_ready_pre_mpc_markers_locked():
                self._pending_rerun_records.append(marker_record)
            self._pending_rerun_records.append(
                {
                    "timestamp": float(timestamp),
                    "source": "infer_complete_event",
                    "action_count": int(action_count) if action_count is not None else None,
                }
            )
            self._writer_cond.notify()

    def add_observation(self, observation: dict[str, Any]) -> None:
        if not self.enabled:
            return
        image_timestamp = float(observation.get("image_timestamp", observation.get("timestamp", 0.0)))
        images = observation.get("images") or {}
        if self._rerun_enabled and images:
            self._enqueue_rerun_images(image_timestamp, images)
        if not self.record_videos:
            return
        for camera_name, image_bytes in images.items():
            if not image_bytes:
                continue
            self._enqueue_video(camera_name, image_timestamp, image_bytes)

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._writer_cond:
            self._writer_stop = True
            self._writer_cond.notify_all()
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join()
        video_files: list[str] = []
        close_error: Exception | None = None
        for camera_name, sink in list(self._video_sinks.items()):
            video_files.append(sink.output_path.name)
            try:
                sink.close()
            except Exception as exc:
                if close_error is None:
                    close_error = exc
            finally:
                self._video_sinks.pop(camera_name, None)
        self._append_jsonl(
            self._stats_records_path(),
            {
                "timestamp": time.time(),
                "dropped_video_frames": int(self._dropped_video_frames),
                "dropped_rerun_frames": int(self._dropped_rerun_frames),
                "rerun_enabled": bool(self._rerun_enabled),
                "rerun_path": self._rerun_path,
                "video_backend": self._video_backend,
                "video_fps": float(self.video_fps),
                "video_files": sorted(video_files),
            },
        )
        if self._writer_error is not None:
            raise self._writer_error
        if close_error is not None:
            raise close_error

    def _rrd_dir(self) -> Path:
        return Path(self.output_dir) / self._rrd_dir_name

    def _pre_mpc_action_records_path(self) -> Path:
        return self._rrd_dir() / "pre_mpc_action_records.jsonl"

    def _raw_pre_mpc_action_records_path(self) -> Path:
        return self._rrd_dir() / "raw_pre_mpc_action_records.jsonl"

    def _post_mpc_action_records_path(self) -> Path:
        return self._rrd_dir() / "post_mpc_action_records.jsonl"

    def _actual_action_records_path(self) -> Path:
        return self._rrd_dir() / "actual_action_records.jsonl"

    def _mpc_action_records_path(self) -> Path:
        return self._rrd_dir() / "mpc_action_records.jsonl"

    def _raw_pre_smooth_action_records_path(self) -> Path:
        return self._rrd_dir() / "raw_pre_smooth_action_records.jsonl"

    def _pre_smooth_action_records_path(self) -> Path:
        return self._rrd_dir() / "pre_smooth_action_records.jsonl"

    def _post_smooth_action_records_path(self) -> Path:
        return self._rrd_dir() / "post_smooth_action_records.jsonl"

    def _smooth_action_records_path(self) -> Path:
        return self._rrd_dir() / "smooth_action_records.jsonl"

    def _request_meta_records_path(self) -> Path:
        return self._rrd_dir() / "request_meta_records.jsonl"

    def _inference_event_records_path(self) -> Path:
        return self._rrd_dir() / "inference_event_records.jsonl"

    def _stats_records_path(self) -> Path:
        return self._rrd_dir() / "recording_stats.jsonl"

    def _initialize_output_files(self) -> None:
        paths = [
            self._actual_action_records_path(),
            self._request_meta_records_path(),
            self._inference_event_records_path(),
            self._stats_records_path(),
        ]
        if self.trajectory_family == "smooth":
            paths.extend(
                [
                    self._raw_pre_smooth_action_records_path(),
                    self._pre_smooth_action_records_path(),
                    self._post_smooth_action_records_path(),
                    self._smooth_action_records_path(),
                ]
            )
        else:
            paths.extend(
                [
                    self._raw_pre_mpc_action_records_path(),
                    self._pre_mpc_action_records_path(),
                    self._post_mpc_action_records_path(),
                    self._mpc_action_records_path(),
                ]
            )
        for path in paths:
            path.write_text("", encoding="utf-8")

    def _pre_action_source(self) -> str:
        return "pre_smooth_action" if self.trajectory_family == "smooth" else "pre_mpc_action"

    def _pre_action_marker_path(self) -> str:
        return "robot/pre_smooth_action_infer_complete" if self.trajectory_family == "smooth" else "robot/pre_mpc_action_infer_complete"

    def _remember_trajectory_record(self, source: str, record: dict[str, Any]) -> list[dict[str, Any]]:
        action = record.get("action")
        if action is None:
            return []
        snapshot = {
            "timestamp": float(record.get("timestamp", 0.0)),
            "action": list(action),
        }
        with self._writer_cond:
            self._latest_trajectory_records[source] = snapshot
            if source != self._pre_action_source():
                return []
            self._pre_mpc_history.append(snapshot)
            while len(self._pre_mpc_history) > 256:
                self._pre_mpc_history.popleft()
            return self._drain_ready_pre_mpc_markers_locked()

    @staticmethod
    def _interpolate_action(
        start_action: list[float],
        end_action: list[float],
        alpha: float,
    ) -> list[float]:
        width = min(len(start_action), len(end_action))
        if width <= 0:
            return []
        alpha = min(1.0, max(0.0, float(alpha)))
        return [
            float(start_action[idx]) + (float(end_action[idx]) - float(start_action[idx])) * alpha
            for idx in range(width)
        ]

    def _interpolate_pre_mpc_marker_action_locked(self, marker_timestamp: float) -> list[float] | None:
        if not self._pre_mpc_history:
            return None
        first = self._pre_mpc_history[0]
        first_ts = float(first["timestamp"])
        if len(self._pre_mpc_history) == 1:
            if marker_timestamp <= first_ts:
                return list(first["action"])
            return None
        if marker_timestamp <= first_ts:
            return list(first["action"])
        prev = self._pre_mpc_history[0]
        for current in list(self._pre_mpc_history)[1:]:
            prev_ts = float(prev["timestamp"])
            cur_ts = float(current["timestamp"])
            if marker_timestamp <= cur_ts:
                if cur_ts <= prev_ts:
                    return list(current["action"])
                alpha = (float(marker_timestamp) - prev_ts) / (cur_ts - prev_ts)
                return self._interpolate_action(
                    list(prev["action"]),
                    list(current["action"]),
                    alpha,
                )
            prev = current
        return None

    def _drain_ready_pre_mpc_markers_locked(self) -> list[dict[str, Any]]:
        if not self._pending_pre_mpc_markers or not self._pre_mpc_history:
            return []
        latest_ts = float(self._pre_mpc_history[-1]["timestamp"])
        ready: list[dict[str, Any]] = []
        pending: deque[dict[str, Any]] = deque()
        while self._pending_pre_mpc_markers:
            marker = self._pending_pre_mpc_markers.popleft()
            marker_ts = float(marker["timestamp"])
            if marker_ts > latest_ts:
                pending.append(marker)
                continue
            marker_action = self._interpolate_pre_mpc_marker_action_locked(marker_ts)
            if marker_action is None:
                pending.append(marker)
                continue
            ready_marker = {
                "timestamp": marker_ts,
                "source": "infer_complete_marker",
                "marker_target": self._pre_action_source(),
                "action": marker_action,
            }
            if marker.get("action_count") is not None:
                ready_marker["action_count"] = int(marker["action_count"])
            ready.append(ready_marker)
        self._pending_pre_mpc_markers = pending
        return ready

    @staticmethod
    def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(item, ensure_ascii=True) + "\n")
            fp.flush()

    def _enqueue_jsonl(self, path: Path, item: dict[str, Any]) -> None:
        with self._writer_cond:
            if self._writer_error is not None:
                return
            self._pending_jsonl_writes.append((path, item))
            self._writer_cond.notify()

    def _enqueue_rerun_record(self, record: dict[str, Any]) -> None:
        with self._writer_cond:
            if self._writer_error is not None or not self._rerun_enabled:
                return
            self._pending_rerun_records.append(record)
            self._writer_cond.notify()

    def _enqueue_rerun_images(self, image_timestamp: float, images: dict[str, Any]) -> None:
        copied_images = {
            str(camera_name): bytes(image_bytes)
            for camera_name, image_bytes in images.items()
            if image_bytes
        }
        if not copied_images:
            return
        with self._writer_cond:
            if self._writer_error is not None or not self._rerun_enabled:
                return
            if len(self._pending_rerun_images) >= max(1, int(self.max_pending_video_frames)):
                self._dropped_rerun_frames += 1
                return
            self._pending_rerun_images.append((float(image_timestamp), copied_images))
            self._writer_cond.notify()

    def _jsonl_writer_thread(self) -> None:
        try:
            while True:
                with self._writer_cond:
                    while (
                        not self._pending_jsonl_writes
                        and not self._pending_rerun_records
                        and not self._pending_rerun_images
                        and not self._pending_video_writes
                        and not self._writer_stop
                    ):
                        self._writer_cond.wait()
                    if (
                        not self._pending_jsonl_writes
                        and not self._pending_rerun_records
                        and not self._pending_rerun_images
                        and not self._pending_video_writes
                        and self._writer_stop
                    ):
                        return
                    jsonl_item = self._pending_jsonl_writes.popleft() if self._pending_jsonl_writes else None
                    rerun_record = (
                        self._pending_rerun_records.popleft()
                        if jsonl_item is None and self._pending_rerun_records
                        else None
                    )
                    rerun_image = (
                        self._pending_rerun_images.popleft()
                        if jsonl_item is None and rerun_record is None and self._pending_rerun_images
                        else None
                    )
                    video_item = (
                        self._pending_video_writes.popleft()
                        if jsonl_item is None and rerun_record is None and rerun_image is None and self._pending_video_writes
                        else None
                    )
                if jsonl_item is not None:
                    path, item = jsonl_item
                    self._append_jsonl(path, item)
                    continue
                if rerun_record is not None:
                    self._log_record_to_rerun(rerun_record)
                    continue
                if rerun_image is not None:
                    image_timestamp, images = rerun_image
                    self._log_images_to_rerun(image_timestamp, images)
                    continue
                if video_item is not None:
                    camera_name, image_timestamp, image_bytes = video_item
                    self._write_video_frame(camera_name, image_timestamp, image_bytes)
        except Exception as exc:
            self._writer_error = exc

    def _video_output_path(self, camera_name: str, use_ffmpeg: bool) -> Path:
        suffix = ".avi" if use_ffmpeg else ".mp4"
        return self._rrd_dir() / f"{camera_name}{suffix}"

    def _create_video_sink(self, camera_name: str) -> _BaseVideoSink:
        if self._ffmpeg_path is not None:
            return _FfmpegMjpegSink(
                ffmpeg_path=self._ffmpeg_path,
                output_path=self._video_output_path(camera_name, use_ffmpeg=True),
                fps=self.video_fps,
            )
        return _OpenCvMp4Sink(
            output_path=self._video_output_path(camera_name, use_ffmpeg=False),
            fps=self.video_fps,
        )

    def _enqueue_video(self, camera_name: str, image_timestamp: float, image_bytes: bytes) -> None:
        with self._writer_cond:
            if self._writer_error is not None:
                return
            if len(self._pending_video_writes) >= max(1, int(self.max_pending_video_frames)):
                self._dropped_video_frames += 1
                return
            self._pending_video_writes.append((camera_name, float(image_timestamp), bytes(image_bytes)))
            self._writer_cond.notify()

    def _write_video_frame(self, camera_name: str, image_timestamp: float, image_bytes: bytes) -> None:
        del image_timestamp
        sink = self._video_sinks.get(camera_name)
        if sink is None:
            sink = self._create_video_sink(camera_name)
            self._video_sinks[camera_name] = sink
        try:
            sink.write_encoded_frame(image_bytes)
        except Exception:
            if not isinstance(sink, _FfmpegMjpegSink):
                raise
            try:
                sink.close()
            except Exception:
                pass
            try:
                sink.output_path.unlink(missing_ok=True)
            except Exception:
                pass
            fallback = _OpenCvMp4Sink(
                output_path=self._video_output_path(camera_name, use_ffmpeg=False),
                fps=self.video_fps,
            )
            self._video_sinks[camera_name] = fallback
            self._video_backend = "mixed"
            fallback.write_encoded_frame(image_bytes)

    def _set_rerun_time(self, timestamp: float | None) -> None:
        if not self._rerun_enabled or timestamp is None:
            return
        rr.set_time_seconds("wall_time", seconds=float(timestamp))

    def _apply_default_blueprint(self) -> None:
        if not self._rerun_enabled or rr is None or rrb is None:
            return
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.TimePanel(state=rrb.PanelState.Hidden),
                auto_views=True,
                auto_layout=True,
            )
        )

    def _ensure_rerun_series(self, base_path: str, width: int) -> None:
        if not self._rerun_enabled or base_path in self._rerun_series_paths:
            return
        names = [f"joint_{idx:02d}" for idx in range(max(0, int(width)))]
        rr.log(base_path, rr.SeriesLines(names=names), static=True)
        self._rerun_series_paths.add(base_path)

    def _ensure_rerun_point_series(self, base_path: str, width: int) -> None:
        if not self._rerun_enabled or base_path in self._rerun_point_series_paths:
            return
        names = [f"joint_{idx:02d}" for idx in range(max(0, int(width)))]
        if hasattr(rr, "SeriesPoints"):
            rr.log(
                base_path,
                rr.SeriesPoints(
                    names=names,
                    markers="circle",
                    marker_sizes=[2.0] * len(names),
                ),
                static=True,
            )
        else:
            rr.log(base_path, rr.SeriesLines(names=names), static=True)
        self._rerun_point_series_paths.add(base_path)

    def _log_vector_scalars(self, base_path: str, values) -> None:
        if not self._rerun_enabled or values is None:
            return
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._ensure_rerun_series(base_path, arr.size)
        rr.log(base_path, rr.Scalars(arr.astype(np.float64, copy=False)))

    def _log_vector_points(self, base_path: str, values) -> None:
        if not self._rerun_enabled or values is None:
            return
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._ensure_rerun_point_series(base_path, arr.size)
        rr.log(base_path, rr.Scalars(arr.astype(np.float64, copy=False)))

    def _log_record_to_rerun(self, record: dict[str, Any]) -> None:
        if not self._rerun_enabled:
            return
        timestamp = float(record.get("timestamp", 0.0))
        source = str(record.get("source", ""))
        self._set_rerun_time(timestamp)
        if source == "infer_complete_event" and hasattr(rr, "TextLog"):
            action_count = record.get("action_count")
            message = "inference complete"
            if action_count is not None:
                message += f" ({int(action_count)} actions)"
            rr.log("events/inference_complete", rr.TextLog(message))
            return
        action = record.get("action")
        if action is None:
            return
        if source == "pre_mpc_action":
            self._log_vector_scalars("robot/pre_mpc_action", action)
        elif source == "raw_pre_mpc_action":
            self._log_vector_scalars("robot/raw_pre_mpc_action", action)
        elif source == "post_mpc_action":
            self._log_vector_scalars("robot/post_mpc_action", action)
        elif source == "raw_pre_smooth_action":
            self._log_vector_scalars("robot/raw_pre_smooth_action", action)
        elif source == "pre_smooth_action":
            self._log_vector_scalars("robot/pre_smooth_action", action)
        elif source == "post_smooth_action":
            self._log_vector_scalars("robot/post_smooth_action", action)
        elif source == "actual_action":
            self._log_vector_scalars("robot/actual_action", action)
        elif source == "infer_complete_marker":
            marker_target = str(record.get("marker_target", ""))
            if marker_target == "pre_mpc_action":
                self._log_vector_points("robot/pre_mpc_action_infer_complete", action)
            elif marker_target == "pre_smooth_action":
                self._log_vector_points("robot/pre_smooth_action_infer_complete", action)

    def _log_images_to_rerun(self, image_timestamp: float, images: dict[str, bytes]) -> None:
        if not self._rerun_enabled:
            return
        self._set_rerun_time(image_timestamp)
        for camera_name, image_bytes in images.items():
            if not image_bytes:
                continue
            if hasattr(rr, "EncodedImage"):
                rr.log(
                    f"cameras/{camera_name}",
                    rr.EncodedImage(contents=bytes(image_bytes), media_type="image/jpeg"),
                )
                continue
            if hasattr(rr, "ImageEncoded"):
                rr.log(
                    f"cameras/{camera_name}",
                    rr.ImageEncoded(contents=bytes(image_bytes)),
                )
                continue
            frame_bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame_bgr is None or frame_bgr.size == 0:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"cameras/{camera_name}", rr.Image(frame_rgb))
