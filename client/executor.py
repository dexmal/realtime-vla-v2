from __future__ import annotations
from abc import ABC
from abc import abstractmethod
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from dataclasses import field
import numpy as np

from robot_io import BaseActuator

try:
    import casadi as ca
except Exception:
    ca = None

try:
    from acados_template import AcadosModel
    from acados_template import AcadosOcp
    from acados_template import AcadosOcpSolver
except Exception:
    AcadosModel = None
    AcadosOcp = None
    AcadosOcpSolver = None

@dataclass
class CommandStamped:
    t: float
    y: np.ndarray

class ReplayEstimator:
    def __init__(self, nq: int, tau: float, max_cmd_hist: int = 500):
        self.nq = nq
        self.tau = tau
        self.cmd_hist = deque(maxlen=max_cmd_hist)
        self.anchor_t = None
        self.anchor_x = None
        self.contact_flag = False
        self.innov_ema = 0.0
        self.innov_alpha = 0.1
        self.contact_threshold = 0.15
        self.contact_hold_steps = 0
        self.contact_hold_min = 8

    def push_command(self, t: float, y: np.ndarray):
        self.cmd_hist.append(CommandStamped(t=t, y=y.copy()))

    def push_observation(self, t_obs: float, x_obs: np.ndarray):
        if self.anchor_t is not None and self.anchor_x is not None:
            x_pred = self._replay(self.anchor_t, self.anchor_x, t_obs)
            innov = np.linalg.norm(x_obs - x_pred) / np.sqrt(self.nq)
            self.innov_ema = (1 - self.innov_alpha) * self.innov_ema + self.innov_alpha * innov
            if self.innov_ema > self.contact_threshold:
                self.contact_flag = True
                self.contact_hold_steps = self.contact_hold_min
        self.anchor_t = t_obs
        self.anchor_x = x_obs.copy()

    def set_contact_params(self, threshold: float, hold_min: int):
        self.contact_threshold = float(threshold)
        self.contact_hold_min = int(hold_min)

    def step_contact_decay(self):
        if self.contact_hold_steps > 0:
            self.contact_hold_steps -= 1
        else:
            self.contact_flag = False

    def estimate_now(self, t_now: float) -> np.ndarray:
        if self.anchor_t is None or self.anchor_x is None:
            raise RuntimeError("No observation anchor yet.")
        if self.contact_flag:
            return self.anchor_x.copy()
        return self._replay(self.anchor_t, self.anchor_x, t_now)

    def _replay(self, t0: float, x0: np.ndarray, t1: float) -> np.ndarray:
        if t1 <= t0:
            return x0.copy()
        cmds = [c for c in self.cmd_hist if c.t <= t1]
        cmds.sort(key=lambda c: c.t)
        x = x0.copy()
        t = t0
        y_active = None
        for c in cmds:
            if c.t <= t0:
                y_active = c.y
            else:
                break
        if y_active is None:
            y_active = x0.copy()
        for c in cmds:
            if c.t <= t0:
                continue
            if c.t > t1:
                break
            dt = c.t - t
            x = y_active + (x - y_active) * np.exp(-dt / self.tau)
            t = c.t
            y_active = c.y
        dt = t1 - t
        x = y_active + (x - y_active) * np.exp(-dt / self.tau)
        return x

class ProgressManager:
    def __init__(self):
        self.progress = 0.0

    def advance(self, alpha: float):
        self.progress += float(alpha)
        if self.progress < 0:
            self.progress = 0.0

    def completed_steps(self) -> int:
        if self.progress <= 0.0:
            return 0
        return int(np.floor(self.progress + 1e-9))

    def consume_steps(self, steps: int):
        if steps <= 0:
            return
        self.progress -= float(steps)
        if self.progress < 0.0:
            self.progress = 0.0

@dataclass
class MPCConfig:
    nq: int
    dt: float = 0.02
    tau: float = 0.15
    N: int = 15
    q_min: np.ndarray = field(default=None)
    q_max: np.ndarray = field(default=None)
    v_max: np.ndarray = field(default=None)
    dy_max: np.ndarray = field(default=None)
    ddy_max: np.ndarray = field(default=None)
    w_track0: float = 10.0
    track_decay: float = 1.0
    w_cmd: float = 0.0
    w_yx: float = 0.0
    w_dy: float = 0.0
    w_ddy: float = 10.0
    contact_v_scale: float = 0.4

    def __post_init__(self):
        nq = self.nq
        if self.q_min is None:
            self.q_min = -np.pi * np.ones(nq)
        if self.q_max is None:
            self.q_max = +np.pi * np.ones(nq)
        if self.v_max is None:
            self.v_max = 50.0 * np.ones(nq)
        if self.dy_max is None:
            self.dy_max = 100.0 * np.ones(nq)
        if self.ddy_max is None:
            self.ddy_max = 100.0 * np.ones(nq)

    def stage_sqrt_w(self, k: int) -> float:
        if k <= 0:
            return 0.0
        w = self.w_track0 * (self.track_decay ** (k - 1))
        return float(np.sqrt(max(w, 1e-8)))

@dataclass
class PlannerOutput:
    y_cmd: np.ndarray
    alpha: float

class AcadosPlanner:
    def __init__(self, cfg: MPCConfig, export_dir: str = "c_generated_code", json_file: str = "acados_ocp.json"):
        self.cfg = cfg
        self.solver = self._build_solver(cfg, export_dir=export_dir, json_file=json_file)
        self.nq = cfg.nq
        self.nx = 3 * self.nq
        self.nu = self.nq
        self.x_aug = np.zeros(self.nx)
        self.initialized = False
        self._contact_mode_cached = None
        self._h_bound_stages = None

    @staticmethod
    def _build_solver(cfg: MPCConfig, export_dir: str = "c_generated_code", json_file: str = "acados_ocp.json"):
        if ca is None or AcadosOcp is None or AcadosOcpSolver is None or AcadosModel is None:
            raise RuntimeError("acados planner dependencies missing. Install casadi and acados_template.")
        nq = cfg.nq
        nx = 3 * nq
        nu = nq
        np_p = nq + 1

        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)
        p = ca.SX.sym("p", np_p)
        x_q = x[0:nq]
        y_prev = x[nq : 2 * nq]
        y_prev2 = x[2 * nq : 3 * nq]
        y_cmd = u[0:nq]
        r_ref = p[0:nq]
        sqrt_w_track = p[-1]

        a = float(np.exp(-cfg.dt / cfg.tau))
        x_next = a * x_q + (1.0 - a) * y_cmd
        y_prev_next = y_cmd
        y_prev2_next = y_prev
        x_next_full = ca.vertcat(x_next, y_prev_next, y_prev2_next)

        model = AcadosModel()
        model.name = "joint_mpc_first_order_discrete"
        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = x_next_full

        e_track = sqrt_w_track * (x_q - r_ref)
        e_cmd = ca.sqrt(cfg.w_cmd) * (y_cmd - r_ref)
        e_yx = ca.sqrt(cfg.w_yx) * (y_cmd - x_q)
        dy = y_cmd - y_prev
        ddy = y_cmd - 2.0 * y_prev + y_prev2
        e_dy = ca.sqrt(cfg.w_dy) * dy
        e_ddy = ca.sqrt(cfg.w_ddy) * ddy
        y_expr = ca.vertcat(e_track, e_cmd, e_yx, e_dy, e_ddy)
        ny = int(y_expr.shape[0])
        y_expr_e = sqrt_w_track * (x_q - r_ref)
        ny_e = int(y_expr_e.shape[0])

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = cfg.N
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.tf = cfg.N * cfg.dt
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver_cond_N = min(5, cfg.N)
        ocp.solver_options.print_level = 0

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr = y_expr
        ocp.model.cost_y_expr_e = y_expr_e
        ocp.cost.W = np.eye(ny)
        ocp.cost.W_e = np.eye(ny_e)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)
        ocp.constraints.lbu = cfg.q_min.copy()
        ocp.constraints.ubu = cfg.q_max.copy()
        ocp.constraints.idxbx_0 = np.arange(nx, dtype=np.int64)
        ocp.constraints.lbx_0 = np.zeros(nx)
        ocp.constraints.ubx_0 = np.zeros(nx)

        h_expr = ca.vertcat((y_cmd - x_q), (y_cmd - y_prev), (y_cmd - 2.0 * y_prev + y_prev2))
        ocp.model.con_h_expr = h_expr
        try:
            ocp.model.con_h_expr_0 = h_expr
        except Exception:
            pass

        e_max = cfg.tau * cfg.v_max
        lh = np.concatenate([-e_max, -cfg.dy_max, -cfg.ddy_max])
        uh = np.concatenate([+e_max, +cfg.dy_max, +cfg.ddy_max])
        ocp.constraints.lh = lh
        ocp.constraints.uh = uh
        try:
            ocp.constraints.lh_0 = lh.copy()
            ocp.constraints.uh_0 = uh.copy()
        except Exception:
            pass

        ocp.parameter_values = np.zeros(np_p)
        ocp.code_export_directory = export_dir
        return AcadosOcpSolver(ocp, json_file=json_file)

    def reset(self, q0: np.ndarray):
        q0 = q0.copy()
        nq = self.nq
        self.x_aug[:] = 0.0
        self.x_aug[0:nq] = q0
        self.x_aug[nq : 2 * nq] = q0
        self.x_aug[2 * nq : 3 * nq] = q0
        self.initialized = True
        self._contact_mode_cached = None

    def _set_stage_h_bounds(self, k: int, lh: np.ndarray, uh: np.ndarray):
        if hasattr(self.solver, "constraints_set"):
            self.solver.constraints_set(k, "lh", lh)
            self.solver.constraints_set(k, "uh", uh)
        else:
            self.solver.set(k, "lh", lh)
            self.solver.set(k, "uh", uh)

    def _set_h_bounds(self, contact_mode: bool):
        scale = float(self.cfg.contact_v_scale) if contact_mode else 1.0
        e_max = self.cfg.tau * (self.cfg.v_max * scale)
        lh = np.concatenate([-e_max, -self.cfg.dy_max, -self.cfg.ddy_max])
        uh = np.concatenate([+e_max, +self.cfg.dy_max, +self.cfg.ddy_max])
        if self._h_bound_stages is None:
            supported_stages = []
            for k in range(self.cfg.N):
                try:
                    self._set_stage_h_bounds(k, lh, uh)
                    supported_stages.append(k)
                except ValueError as exc:
                    if "mismatching dimension" not in str(exc):
                        raise
            self._h_bound_stages = tuple(supported_stages)
            if not self._h_bound_stages:
                print("[executor] RTI planner has no writable h constraints; skipping online h-bound updates.")
                return
            return
        for k in self._h_bound_stages:
            self._set_stage_h_bounds(k, lh, uh)

    def _rate_limit_fallback(self, q_hat: np.ndarray, y_des: np.ndarray, contact_mode: bool) -> np.ndarray:
        nq = self.nq
        cfg = self.cfg
        x = q_hat.copy()
        y_prev = self.x_aug[nq : 2 * nq].copy()
        y_prev2 = self.x_aug[2 * nq : 3 * nq].copy()
        y = np.clip(y_des, cfg.q_min, cfg.q_max)
        dy = y - y_prev
        dy = np.clip(dy, -cfg.dy_max, cfg.dy_max)
        y = y_prev + dy
        d2 = y - 2.0 * y_prev + y_prev2
        d2 = np.clip(d2, -cfg.ddy_max, cfg.ddy_max)
        y = 2.0 * y_prev - y_prev2 + d2
        scale = float(cfg.contact_v_scale) if contact_mode else 1.0
        e_max = cfg.tau * (cfg.v_max * scale)
        e = y - x
        e = np.clip(e, -e_max, e_max)
        y = x + e
        y = np.clip(y, cfg.q_min, cfg.q_max)
        return y

    def solve(self, q_hat: np.ndarray, ai_future: np.ndarray, contact_mode: bool = False) -> PlannerOutput:
        if not self.initialized:
            self.reset(q_hat)
        nq = self.nq
        cfg = self.cfg
        ai_future = np.asarray(ai_future, dtype=np.float32)
        if ai_future.ndim != 2:
            raise ValueError("ai_future must be 2D array.")
        if ai_future.shape[1] != nq and ai_future.shape[0] == nq:
            ai_future = ai_future.T
        if ai_future.shape[1] != nq:
            raise ValueError(f"ai_future shape mismatch: got {ai_future.shape}, expected (*, {nq})")
        if ai_future.shape[0] < cfg.N:
            pad = np.repeat(ai_future[-1:, :], cfg.N - ai_future.shape[0], axis=0)
            ai_future = np.vstack([ai_future, pad])
        else:
            ai_future = ai_future[: cfg.N, :]
        if contact_mode:
            ai_use = np.repeat(q_hat.reshape(1, -1), cfg.N, axis=0)
        else:
            ai_use = ai_future
        if self._contact_mode_cached is None or bool(contact_mode) != bool(self._contact_mode_cached):
            self._set_h_bounds(contact_mode=bool(contact_mode))
            self._contact_mode_cached = bool(contact_mode)
        self.x_aug[0:nq] = q_hat.copy()
        self.solver.set(0, "lbx", self.x_aug)
        self.solver.set(0, "ubx", self.x_aug)
        for k in range(cfg.N):
            r_k = ai_use[k]
            p_k = np.concatenate([r_k, [cfg.stage_sqrt_w(k)]])
            self.solver.set(k, "p", p_k)
        p_e = np.concatenate([ai_use[-1], [cfg.stage_sqrt_w(cfg.N)]])
        self.solver.set(cfg.N, "p", p_e)
        status = int(self.solver.solve())
        if status != 0:
            y_cmd = self._rate_limit_fallback(q_hat=q_hat, y_des=q_hat, contact_mode=contact_mode)
            alpha = 0.0
            self.x_aug[2 * nq : 3 * nq] = self.x_aug[nq : 2 * nq].copy()
            self.x_aug[nq : 2 * nq] = y_cmd.copy()
            self.x_aug[0:nq] = q_hat.copy()
            return PlannerOutput(y_cmd=y_cmd, alpha=alpha)
        u0 = self.solver.get(0, "u")
        y_cmd = np.asarray(u0[0:nq]).copy()
        x1 = self.solver.get(1, "x").copy()
        self.x_aug = x1
        alpha = 0.0 if contact_mode else 1.0
        return PlannerOutput(y_cmd=y_cmd, alpha=float(alpha))

def _find_nearest_history_index(history: list[dict], target_time: float) -> int:
    if not history:
        return 0
    best_idx = 0
    best_delta = abs(float(history[0]["timestamp"]) - float(target_time))
    for idx in range(1, len(history)):
        delta = abs(float(history[idx]["timestamp"]) - float(target_time))
        if delta < best_delta:
            best_delta = delta
            best_idx = idx
    return best_idx


def _expand_optional_array(
    values: list[float] | tuple[float, ...] | None,
    size: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    parsed = [float(v) for v in values]
    if len(parsed) == 0:
        return None
    if len(parsed) == 1:
        return np.full((size,), parsed[0], dtype=np.float64)
    if len(parsed) != size:
        raise ValueError(f"{name} expects empty, 1 or {size} values, got {len(parsed)}")
    return np.asarray(parsed, dtype=np.float64)


def _predict_steps_static(latency_s: float, interval_ms: float | None) -> int:
    if interval_ms is None or interval_ms <= 0:
        return 1
    interval_s = float(interval_ms) / 1000.0
    if interval_s <= 0:
        return 1
    return max(1, int(math.ceil(float(latency_s) / interval_s)))


def _gripper_heartbeat_lookahead_index(
    interval_ms: float,
    lookahead_ms: float | None = 90.0,
) -> int:
    if lookahead_ms is None:
        return 0
    lookahead_ms = float(lookahead_ms)
    if lookahead_ms <= 0:
        return 0
    return max(0, int(_predict_steps_static(lookahead_ms / 1000.0, interval_ms)))


def _apply_gripper_heartbeat_lookahead(
    action: list[float] | None,
    future_queue: list[list[float]],
    lookahead_idx: int,
    lookahead_dims: tuple[int, ...] = (6, 13),
) -> list[float] | None:
    if action is None:
        return None
    adjusted = list(action)
    if not future_queue:
        return adjusted
    source_idx = min(max(int(lookahead_idx), 0), len(future_queue) - 1)
    source_action = future_queue[source_idx]
    for dim in lookahead_dims:
        if dim < len(adjusted) and dim < len(source_action):
            adjusted[dim] = float(source_action[dim])
    return adjusted


@dataclass(frozen=True)
class ActionTransform:
    fixed_pairs: tuple[tuple[int, float], ...] = ()

    def apply(self, action: list[float] | np.ndarray) -> list[float]:
        arr = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        for dim, value in self.fixed_pairs:
            if 0 <= int(dim) < arr.size:
                arr[int(dim)] = float(value)
        return arr.tolist()


def _build_action_transform(
    fixed_dims: list[int] | tuple[int, ...] | None,
    fixed_values: list[float] | tuple[float, ...] | None,
    *,
    name: str,
) -> ActionTransform:
    dims = [int(v) for v in (fixed_dims or ())]
    values = [float(v) for v in (fixed_values or ())]
    if len(dims) != len(values):
        raise ValueError(f"{name} expects fixed dims/values with the same length")
    return ActionTransform(
        fixed_pairs=tuple((dim, value) for dim, value in zip(dims, values)),
    )


class ForwardTracker:
    def __init__(self, init_state: np.ndarray, alpha: float = 1.0, delay_cnt: int = 5, lead_s: float = 0.15):
        self.delay_cnt = max(1, int(delay_cnt))
        self.alpha = float(alpha)
        self.lead_s = max(0.0, float(lead_s))
        init_state = np.asarray(init_state, dtype=np.float32).reshape(-1)
        self.send_history = np.zeros((self.delay_cnt, init_state.shape[0]), dtype=np.float32)
        for idx in range(self.delay_cnt):
            self.send_history[idx] = init_state

    def reset(self, init_state: np.ndarray) -> None:
        init_state = np.asarray(init_state, dtype=np.float32).reshape(-1)
        for idx in range(self.send_history.shape[0]):
            self.send_history[idx] = init_state

    def track(
        self,
        new_observed: np.ndarray,
        new_target: np.ndarray,
        new_vel: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed = np.asarray(new_observed, dtype=np.float32).reshape(-1)
        target = np.asarray(new_target, dtype=np.float32).reshape(-1)
        vel = np.asarray(new_vel, dtype=np.float32).reshape(-1)
        lead_s = self.lead_s * self.alpha
        if lead_s > 0.0:
            for send_state in self.send_history:
                velocity = (send_state - observed) / lead_s
                observed = observed + velocity * float(dt_s)
        new_send = target + vel * lead_s
        for dim in (6, 13):
            if dim < target.size and dim < new_send.size:
                new_send[dim] = target[dim]
        self.send_history = np.roll(self.send_history, -1, axis=0)
        self.send_history[-1] = new_send
        return observed, new_send


def _compute_speed_limited_interval_ms(
    action: list[float] | np.ndarray | None,
    next_action: list[float] | np.ndarray | None,
    min_interval_ms: float,
    speed_limit_per_s: float | None,
    speed_limit_dims: tuple[int, ...] = (),
) -> float:
    min_interval_ms = max(0.0, float(min_interval_ms or 0.0))
    if speed_limit_per_s is None or speed_limit_per_s <= 0:
        return float(min_interval_ms)
    if action is None or next_action is None:
        return float(min_interval_ms)
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    next_arr = np.asarray(next_action, dtype=np.float32).reshape(-1)
    dims = tuple(int(dim) for dim in speed_limit_dims if 0 <= int(dim) < action_arr.size and int(dim) < next_arr.size)
    if dims:
        deltas = np.abs(next_arr[list(dims)] - action_arr[list(dims)])
    else:
        max_len = min(action_arr.size, next_arr.size)
        if max_len <= 0:
            return float(min_interval_ms)
        deltas = np.abs(next_arr[:max_len] - action_arr[:max_len])
    max_delta = float(np.max(deltas)) if deltas.size > 0 else 0.0
    if not np.isfinite(max_delta) or max_delta <= 0.0:
        return float(min_interval_ms)
    required_ms = (max_delta / float(speed_limit_per_s)) * 1000.0
    return float(required_ms) if required_ms > min_interval_ms else float(min_interval_ms)


def _plan_timeline_times(
    history: list[dict],
    timeline: list[list[float]],
    base_time: float | None,
    min_interval_ms: float,
    speed_limit_per_s: float,
    speed_limit_dims: tuple[int, ...],
) -> list[float]:
    times: list[float] = []
    prev_action = None
    prev_time = float(base_time) if base_time is not None else time.time()
    if history:
        for item in history:
            times.append(float(item["timestamp"]))
        prev_action = history[-1]["action"]
        prev_time = times[-1] if times else prev_time
    for idx in range(len(history), len(timeline)):
        action = timeline[idx]
        if prev_action is None:
            cur_time = prev_time
        else:
            interval_ms = _compute_speed_limited_interval_ms(
                prev_action,
                action,
                min_interval_ms,
                speed_limit_per_s,
                speed_limit_dims,
            )
            cur_time = prev_time + interval_ms / 1000.0
        times.append(float(cur_time))
        prev_action = action
        prev_time = cur_time
    return times


def _compute_savgol_weights(window_length: int, polyorder: int) -> np.ndarray | None:
    if window_length <= 1 or window_length % 2 == 0:
        return None
    if polyorder < 0 or polyorder >= window_length:
        return None
    half = window_length // 2
    positions = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(positions, polyorder + 1, increasing=True)
    coeffs = np.linalg.pinv(design)[0]
    return coeffs.astype(np.float32)


def _ensure_action_shape(
    action: list[float] | np.ndarray | None,
    reference: np.ndarray | None,
) -> np.ndarray | None:
    if action is None:
        return None
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if reference is None or arr.shape == reference.shape:
        return arr
    return reference.copy()


def _compute_future_servo_actions(
    now_time: float,
    action: np.ndarray | None,
    next_action: np.ndarray | None,
    start_time: float,
    current_interval_ms: float,
    future_heartbeat_actions: list[list[float]],
    servo_interval_s: float,
    min_interval_ms: float,
    speed_limit_per_s: float,
    speed_limit_dims: tuple[int, ...],
    count: int,
) -> list[np.ndarray]:
    if action is None or count <= 0:
        return []
    action_arr = _ensure_action_shape(action, None)
    if action_arr is None:
        return []
    next_arr = _ensure_action_shape(next_action, action_arr)
    if next_arr is None:
        next_arr = action_arr.copy()
    if start_time <= 0:
        start_time = now_time
    future_list: list[np.ndarray] = []
    for item in future_heartbeat_actions:
        arr = _ensure_action_shape(item, action_arr)
        if arr is not None:
            future_list.append(arr)
    if future_list and np.array_equal(future_list[0], next_arr):
        future_list = future_list[1:]
    key_actions = [action_arr, next_arr] + future_list
    times = [float(start_time)]
    interval_ms = current_interval_ms
    if interval_ms <= 0:
        interval_ms = _compute_speed_limited_interval_ms(
            action_arr,
            next_arr,
            min_interval_ms,
            speed_limit_per_s,
            speed_limit_dims,
        )
    if interval_ms <= 0:
        interval_ms = min_interval_ms if min_interval_ms > 0 else 0.0
    times.append(times[-1] + interval_ms / 1000.0)
    prev_action = next_arr
    for action_next in key_actions[2:]:
        interval_ms = _compute_speed_limited_interval_ms(
            prev_action,
            action_next,
            min_interval_ms,
            speed_limit_per_s,
            speed_limit_dims,
        )
        if interval_ms <= 0:
            interval_ms = min_interval_ms if min_interval_ms > 0 else 0.0
        times.append(times[-1] + interval_ms / 1000.0)
        prev_action = action_next
    future_servo: list[np.ndarray] = []
    segment_idx = 0
    for step in range(1, count + 1):
        target_time = now_time + servo_interval_s * step
        while segment_idx + 1 < len(times) and target_time > times[segment_idx + 1]:
            segment_idx += 1
        if segment_idx + 1 >= len(times):
            future_servo.append(key_actions[-1].copy())
            continue
        t0 = times[segment_idx]
        t1 = times[segment_idx + 1]
        alpha = 0.0 if t1 <= t0 else (target_time - t0) / (t1 - t0)
        alpha = min(1.0, max(0.0, float(alpha)))
        a0 = key_actions[segment_idx]
        a1 = key_actions[segment_idx + 1]
        future_servo.append(a0 + (a1 - a0) * alpha)
    return future_servo


def _savgol_smooth_action(
    action: np.ndarray | None,
    history_actions: list[np.ndarray],
    future_actions: list[np.ndarray],
    weights: np.ndarray | None,
    window_length: int,
) -> np.ndarray | None:
    if action is None:
        return None
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if weights is None or window_length <= 1:
        return action_arr
    half = window_length // 2
    past = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history_actions][-half:]
    future = [np.asarray(item, dtype=np.float32).reshape(-1) for item in future_actions][:half]
    if len(past) < half:
        pad_val = past[0] if past else action_arr
        past = [pad_val.copy() for _ in range(half - len(past))] + past
    if len(future) < half:
        pad_val = future[-1] if future else action_arr
        future = future + [pad_val.copy() for _ in range(half - len(future))]
    window = np.vstack(past + [action_arr] + future)
    if window.shape[0] != window_length or window.shape[1] != action_arr.shape[0]:
        return action_arr
    return weights @ window

class BaseExecutor(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, actuator: BaseActuator):
        raise NotImplementedError

    @abstractmethod
    def get_pending_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_control_dt_s(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def normalize_infer_actions(self, action_list) -> list[list[float]]:
        raise NotImplementedError

    def has_control_thread(self) -> bool:
        return False

    def control_step(
        self,
        _current_state: list[float] | None,
        _state_timestamp: float | None = None,
    ) -> tuple[float, list[dict]]:
        return 0.01, []

@dataclass
class RawActionExecutor(BaseExecutor):
    enable_init_action: bool = False
    init_action: list[float] = field(default_factory=list)
    init_steps: int = 100
    init_sleep_s: float = 0.01
    control_dt_s: float = 0.02
    obs_image_delay_ms: float = 55.0
    state_delay_s: float = 0.05
    max_prefill_states: int = 15
    heartbeat_history_len: int = 200
    infer_fixed_dims: list[int] = field(default_factory=list)
    infer_fixed_values: list[float] = field(default_factory=list)
    command_fixed_dims: list[int] = field(default_factory=list)
    command_fixed_values: list[float] = field(default_factory=list)
    action_interval_ms: float = 10.0
    action_speed_limit_per_s: float = 0.0
    action_speed_limit_dims: tuple[int, ...] = ()
    enable_servo_interpolation: bool = False
    servo_interval_ms: float = 10.0
    savgol_window_length: int = 1
    savgol_polyorder: int = 3
    forward_track_alpha: float = 1.0
    forward_track_delay_cnt: int = 5
    forward_track_lead_s: float = 0.15
    actuator: BaseActuator | None = None

    @classmethod
    def from_config(cls, cfg, actuator: BaseActuator):
        ex = cfg.executor
        return cls(
            enable_init_action=bool(ex.enable_init_action),
            init_action=[float(v) for v in ex.init_action],
            init_steps=int(ex.init_steps),
            init_sleep_s=float(ex.init_sleep_s),
            control_dt_s=float(ex.control_dt_s),
            obs_image_delay_ms=float(ex.obs_image_delay_ms),
            state_delay_s=float(ex.state_delay_s),
            max_prefill_states=int(ex.max_prefill_states),
            heartbeat_history_len=int(ex.heartbeat_history_len),
            infer_fixed_dims=[int(v) for v in ex.infer_fixed_dims],
            infer_fixed_values=[float(v) for v in ex.infer_fixed_values],
            command_fixed_dims=[int(v) for v in ex.command_fixed_dims],
            command_fixed_values=[float(v) for v in ex.command_fixed_values],
            action_interval_ms=float(ex.action_interval_ms),
            action_speed_limit_per_s=float(ex.action_speed_limit_per_s),
            action_speed_limit_dims=tuple(int(v) for v in ex.action_speed_limit_dims),
            enable_servo_interpolation=bool(ex.enable_servo_interpolation),
            servo_interval_ms=float(ex.servo_interval_ms),
            savgol_window_length=int(ex.savgol_window_length),
            savgol_polyorder=int(ex.savgol_polyorder),
            forward_track_alpha=float(ex.forward_track_alpha),
            forward_track_delay_cnt=int(ex.forward_track_delay_cnt),
            forward_track_lead_s=float(ex.forward_track_lead_s),
            actuator=actuator,
        )

    def __post_init__(self) -> None:
        self._actuator: BaseActuator = self.actuator
        self._action_queue: deque[list[float]] = deque()
        self._raw_action_queue: deque[list[float]] = deque()
        self._action_queue_lock = threading.Lock()

        self._control_dt_s = max(1e-3, float(self.control_dt_s))
        self._obs_image_delay_s = float(self.obs_image_delay_ms) / 1000.0
        self._max_prefill_states = int(self.max_prefill_states)
        self._min_interval_ms = float(self.action_interval_ms)
        if self._min_interval_ms <= 0.0:
            self._min_interval_ms = self._control_dt_s * 1000.0
        self._speed_limit_per_s = float(self.action_speed_limit_per_s)
        self._speed_limit_dims = tuple(int(v) for v in self.action_speed_limit_dims)
        self._enable_servo = bool(self.enable_servo_interpolation)
        self._servo_interval_s = float(self.servo_interval_ms) / 1000.0
        if self._servo_interval_s <= 0.0:
            self._servo_interval_s = self._control_dt_s
        self._savgol_weights = _compute_savgol_weights(
            int(self.savgol_window_length),
            int(self.savgol_polyorder),
        )
        self._infer_transform = _build_action_transform(
            self.infer_fixed_dims,
            self.infer_fixed_values,
            name="raw_action infer transform",
        )
        self._command_transform = _build_action_transform(
            self.command_fixed_dims,
            self.command_fixed_values,
            name="raw_action command transform",
        )

        self._exec_step_count = 0
        self._last_executed_action: list[float] | None = None
        self._heartbeat_action_history: deque[dict] = deque(maxlen=max(1, int(self.heartbeat_history_len)))
        self._heartbeat_action: list[float] | None = None
        self._heartbeat_next_action: list[float] | None = None
        self._raw_heartbeat_action: list[float] | None = None
        self._raw_heartbeat_next_action: list[float] | None = None
        self._heartbeat_started_at = 0.0
        self._heartbeat_interval_ms = float(self._min_interval_ms)
        self._servo_action_history: deque[np.ndarray] = deque(
            maxlen=max(int(self.savgol_window_length) * 4, 32)
        )
        self._last_targets: list[np.ndarray] = []
        self._tracker: ForwardTracker | None = None

    def normalize_infer_actions(self, action_list) -> list[list[float]]:
        if not isinstance(action_list, (list, tuple)):
            raise TypeError(f"invalid action_list type: {type(action_list)}")
        return [self._infer_transform.apply(action) for action in action_list]

    def _predict_steps(
        self,
        latency_s: float,
        request_time: float | None = None,
        history: list[dict] | None = None,
        future_queue: list[list[float]] | None = None,
    ) -> int:
        if request_time is None or future_queue is None:
            interval_ms = self._heartbeat_interval_ms if self._heartbeat_interval_ms > 0 else self._min_interval_ms
            return _predict_steps_static(latency_s, interval_ms)
        if history is None:
            history = []
        if not future_queue:
            return _predict_steps_static(latency_s, self._min_interval_ms)
        target_time = float(request_time) + max(0.0, float(latency_s))
        prev_action = history[-1]["action"] if history else None
        prev_time = float(history[-1]["timestamp"]) if history else float(request_time)
        future_times: list[float] = []
        for action in future_queue:
            if prev_action is None:
                cur_time = prev_time
            else:
                interval_ms = _compute_speed_limited_interval_ms(
                    prev_action,
                    action,
                    self._min_interval_ms,
                    self._speed_limit_per_s,
                    self._speed_limit_dims,
                )
                cur_time = prev_time + interval_ms / 1000.0
            future_times.append(float(cur_time))
            prev_action = action
            prev_time = cur_time
        predicted_idx = None
        for idx, cur_time in enumerate(future_times):
            if cur_time >= target_time:
                predicted_idx = idx
                break
        if predicted_idx is None:
            interval_ms = _compute_speed_limited_interval_ms(
                prev_action,
                prev_action,
                self._min_interval_ms,
                self._speed_limit_per_s,
                self._speed_limit_dims,
            )
            interval_s = interval_ms / 1000.0
            if interval_s <= 0.0:
                predicted_idx = len(future_times) - 1
            else:
                extra = int(math.ceil((target_time - future_times[-1]) / interval_s))
                if extra < 0:
                    extra = 0
                predicted_idx = len(future_times) - 1 + extra
        if predicted_idx < 1:
            predicted_idx = 1
        return int(predicted_idx)

    def _build_state_trajectory(
        self,
        predicted_idx: int,
        anchor_time: float | None,
        pad_action: list[float] | None,
        history: list[dict],
        future_queue: list[list[float]],
    ) -> tuple[list[list[float]], int, int, int, list[list[float]]]:
        history_len = len(history)
        history_actions = [list(item["action"]) for item in history]
        timeline = history_actions + [list(action) for action in future_queue]
        predicted_timeline_idx = history_len + int(predicted_idx)
        if predicted_timeline_idx >= len(timeline):
            if pad_action is None:
                pad_action = list(timeline[-1]) if timeline else None
            if pad_action is not None:
                while len(timeline) <= predicted_timeline_idx:
                    timeline.append(list(pad_action))

        start_idx = 0
        if history and anchor_time is not None:
            start_idx = _find_nearest_history_index(history, anchor_time)
        anchor_timeline_idx = predicted_timeline_idx
        if anchor_timeline_idx < start_idx:
            anchor_timeline_idx = start_idx
        if self._max_prefill_states > 0 and anchor_timeline_idx - start_idx + 1 > self._max_prefill_states:
            start_idx += anchor_timeline_idx - start_idx + 1 - self._max_prefill_states
        trajectory = timeline[start_idx : anchor_timeline_idx + 1]
        anchor_idx_in_future = anchor_timeline_idx - history_len
        return trajectory, anchor_idx_in_future, start_idx, anchor_timeline_idx, timeline

    def get_pending_count(self) -> int:
        with self._action_queue_lock:
            return len(self._action_queue)

    def get_control_dt_s(self) -> float:
        return float(self._control_dt_s)

    def has_control_thread(self) -> bool:
        return bool(self._enable_servo)

    def heartbeat_step(
        self, current_state: list[float] | None, state_timestamp: float | None = None
    ) -> tuple[float, list[dict]]:
        records: list[dict] = []
        interval_ms = self._min_interval_ms
        if current_state is not None:
            ts = float(state_timestamp if state_timestamp is not None else time.time())
            records.append({"timestamp": ts, "action": list(current_state), "source": "state"})

        with self._action_queue_lock:
            if self._action_queue:
                action = list(self._action_queue.popleft())
            else:
                action = (
                    list(self._last_executed_action)
                    if self._last_executed_action is not None
                    else (list(current_state) if current_state is not None else None)
                )
            if self._raw_action_queue:
                raw_action = list(self._raw_action_queue.popleft())
            else:
                raw_action = list(action) if action is not None else None
            if action is not None:
                heartbeat_time = time.time()
                next_action = list(self._action_queue[0]) if self._action_queue else list(action)
                raw_next_action = (
                    list(self._raw_action_queue[0])
                    if self._raw_action_queue
                    else (list(next_action) if next_action is not None else None)
                )
                interval_ms = _compute_speed_limited_interval_ms(
                    action,
                    next_action,
                    self._min_interval_ms,
                    self._speed_limit_per_s,
                    self._speed_limit_dims,
                )
                self._last_executed_action = list(action)
                self._heartbeat_action = list(action)
                self._heartbeat_next_action = list(next_action)
                self._raw_heartbeat_action = list(raw_action) if raw_action is not None else None
                self._raw_heartbeat_next_action = list(raw_next_action) if raw_next_action is not None else None
                self._heartbeat_started_at = heartbeat_time
                self._heartbeat_interval_ms = float(interval_ms)
                self._heartbeat_action_history.append({"timestamp": heartbeat_time, "action": list(action)})
                self._exec_step_count += 1
                if not self._enable_servo:
                    post_action = self._command_transform.apply(action)
                    self.apply_action(np.asarray(action, dtype=np.float32).reshape(-1))
                    if raw_action is not None:
                        records.append(
                            {
                                "timestamp": heartbeat_time,
                                "action": list(raw_action),
                                "source": "raw_pre_smooth_action",
                            }
                        )
                    records.append(
                        {
                            "timestamp": heartbeat_time,
                            "action": list(action),
                            "source": "pre_smooth_action",
                        }
                    )
                    records.append(
                        {
                            "timestamp": heartbeat_time,
                            "action": list(post_action),
                            "source": "post_smooth_action",
                        }
                    )
        return float(interval_ms) / 1000.0, records

    def prepare_infer_payload(
        self,
        current_state: list[float] | None,
        images: dict | None,
        image_timestamp: float | None,
        infer_latency_s: float,
    ) -> tuple[dict | None, dict | None]:
        if current_state is None or not images:
            return None, None
        if image_timestamp is None:
            image_timestamp = time.time()
        request_time = time.time()
        with self._action_queue_lock:
            if not self._action_queue:
                self._action_queue.append(list(current_state))
                if self._last_executed_action is None:
                    self._last_executed_action = list(current_state)
            if not self._raw_action_queue:
                self._raw_action_queue.append(list(current_state))
            history_snapshot = [
                {"timestamp": float(item["timestamp"]), "action": list(item["action"])}
                for item in self._heartbeat_action_history
            ]
            future_queue_snapshot = [list(action) for action in self._action_queue]
            predicted_steps = self._predict_steps(
                infer_latency_s,
                request_time=request_time,
                history=history_snapshot,
                future_queue=future_queue_snapshot,
            )
            queue_len_before_pad = len(self._action_queue)
            pad_action = list(self._action_queue[-1])
            padded = predicted_steps >= queue_len_before_pad
            while len(self._action_queue) <= predicted_steps:
                self._action_queue.append(list(pad_action))
            anchor_time = float(image_timestamp) - self._obs_image_delay_s
            (
                state_trajectory,
                anchor_idx_in_future,
                start_idx,
                anchor_timeline_idx,
                timeline,
            ) = self._build_state_trajectory(
                predicted_steps, anchor_time, pad_action, history_snapshot, future_queue_snapshot
            )
            timeline_times = _plan_timeline_times(
                history_snapshot,
                timeline,
                request_time,
                self._min_interval_ms,
                self._speed_limit_per_s,
                self._speed_limit_dims,
            )
            prefill_end = min(anchor_timeline_idx + 1, len(timeline_times))
            prefill_times = timeline_times[start_idx:prefill_end]
            prefill_actions = [list(action) for action in state_trajectory]
            if prefill_times and len(prefill_actions) > len(prefill_times):
                prefill_actions = prefill_actions[: len(prefill_times)]
            context = {
                "exec_step_at_request": self._exec_step_count,
                "predicted_idx": anchor_idx_in_future,
                "predicted_steps": predicted_steps,
                "padded": padded,
                "pad_action": pad_action,
                "image_timestamp": image_timestamp,
                "prefill_actions": prefill_actions,
                "prefill_times": prefill_times,
                "request_time": request_time,
            }
        payload = {
            "images": images,
            "action": state_trajectory,
            "state_delta": predicted_steps,
            "timestamp": request_time,
        }
        return payload, context

    def on_infer_actions(
        self,
        action_list: list[list[float]],
        context: dict,
        raw_action_list: list[list[float]] | None = None,
    ) -> None:
        normalized_actions = [list(action) for action in action_list]
        if not normalized_actions:
            return
        normalized_raw_actions = (
            [list(action) for action in raw_action_list]
            if raw_action_list
            else [list(action) for action in normalized_actions]
        )
        predicted_idx = int(context["predicted_idx"])
        exec_step_at_request = int(context["exec_step_at_request"])
        with self._action_queue_lock:
            executed_since_request = self._exec_step_count - exec_step_at_request
            if executed_since_request < 0:
                executed_since_request = 0

            anchor_idx_current = predicted_idx - executed_since_request
            queue_list = [list(action) for action in self._action_queue]
            if anchor_idx_current >= 0:
                if context.get("padded"):
                    pad_action = context.get("pad_action")
                    if pad_action is not None and queue_list:
                        tail_start = len(queue_list)
                        for idx in range(len(queue_list) - 1, -1, -1):
                            if queue_list[idx] == pad_action:
                                tail_start = idx
                            else:
                                break
                        if tail_start <= anchor_idx_current:
                            anchor_idx_current = tail_start
                overwrite_start_idx = anchor_idx_current + 1
                prefix = queue_list[:overwrite_start_idx]
                self._action_queue = deque(prefix + normalized_actions)
                raw_queue_list = [list(action) for action in self._raw_action_queue]
                raw_prefix = raw_queue_list[:overwrite_start_idx]
                self._raw_action_queue = deque(raw_prefix + normalized_raw_actions)
            else:
                executed_after_anchor = executed_since_request - (predicted_idx + 1)
                if executed_after_anchor < 0:
                    executed_after_anchor = 0
                if executed_after_anchor >= len(normalized_actions):
                    executed_after_anchor = len(normalized_actions) - 1
                self._action_queue = deque(normalized_actions[executed_after_anchor:])
                if executed_after_anchor >= len(normalized_raw_actions):
                    executed_after_anchor = len(normalized_raw_actions) - 1
                self._raw_action_queue = deque(normalized_raw_actions[executed_after_anchor:])

    def control_step(
        self,
        current_state: list[float] | None,
        _state_timestamp: float | None = None,
    ) -> tuple[float, list[dict]]:
        interval_s = self._servo_interval_s if self._servo_interval_s > 0.0 else self._control_dt_s
        if not self._enable_servo:
            return interval_s, []
        loop_time = time.time()
        observed = None if current_state is None else list(current_state)
        with self._action_queue_lock:
            action = list(self._heartbeat_action) if self._heartbeat_action is not None else None
            next_action = list(self._heartbeat_next_action) if self._heartbeat_next_action is not None else None
            raw_action = list(self._raw_heartbeat_action) if self._raw_heartbeat_action is not None else None
            raw_next_action = (
                list(self._raw_heartbeat_next_action)
                if self._raw_heartbeat_next_action is not None
                else None
            )
            start_time = float(self._heartbeat_started_at)
            heartbeat_interval_ms = float(self._heartbeat_interval_ms)
            future_queue = [list(item) for item in self._action_queue]
        if action is None:
            action = observed
        if action is None:
            return interval_s, []
        if next_action is None:
            next_action = action
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        next_arr = np.asarray(next_action, dtype=np.float32).reshape(-1)
        raw_action_arr = _ensure_action_shape(raw_action, action_arr)
        if raw_action_arr is None:
            raw_action_arr = action_arr.copy()
        raw_next_arr = _ensure_action_shape(raw_next_action, raw_action_arr)
        if raw_next_arr is None:
            raw_next_arr = raw_action_arr.copy()
        if action_arr.shape != next_arr.shape:
            next_arr = action_arr.copy()
        if raw_action_arr.shape != raw_next_arr.shape:
            raw_next_arr = raw_action_arr.copy()
        if start_time > 0.0 and heartbeat_interval_ms > 0.0:
            alpha = (loop_time - start_time) / (heartbeat_interval_ms / 1000.0)
            alpha = min(1.0, max(0.0, float(alpha)))
        else:
            alpha = 0.0
        raw_pre_smooth_action = raw_action_arr + (raw_next_arr - raw_action_arr) * alpha
        target_raw = action_arr + (next_arr - action_arr) * alpha
        history_actions: list[np.ndarray] = []
        future_actions: list[np.ndarray] = []
        if self._savgol_weights is not None:
            for item in self._servo_action_history:
                arr = np.asarray(item, dtype=np.float32).reshape(-1)
                if arr.shape == action_arr.shape:
                    history_actions.append(arr)
            half = int(self.savgol_window_length) // 2
            if half > 0:
                future_actions = _compute_future_servo_actions(
                    loop_time,
                    action_arr,
                    next_arr,
                    start_time,
                    heartbeat_interval_ms,
                    future_queue,
                    interval_s,
                    self._min_interval_ms,
                    self._speed_limit_per_s,
                    self._speed_limit_dims,
                    half,
                )
        smoothed_action = _savgol_smooth_action(
            target_raw,
            history_actions,
            future_actions,
            self._savgol_weights,
            int(self.savgol_window_length),
        )
        if smoothed_action is None:
            smoothed_action = target_raw
        target_action = np.asarray(smoothed_action, dtype=np.float32).reshape(-1)
        observed_arr = (
            np.asarray(observed, dtype=np.float32).reshape(-1)
            if observed is not None
            else target_action.copy()
        )
        if observed_arr.shape != target_action.shape:
            observed_arr = target_action.copy()
        self._servo_action_history.append(target_raw.copy())
        if self._tracker is None or self._tracker.send_history.shape[1] != target_action.shape[0]:
            self._tracker = ForwardTracker(
                init_state=observed_arr,
                alpha=self.forward_track_alpha,
                delay_cnt=self.forward_track_delay_cnt,
                lead_s=self.forward_track_lead_s,
            )
            self._last_targets = []
        if self._last_targets:
            denom = interval_s * len(self._last_targets)
            vel_target = np.zeros_like(target_action) if denom <= 0.0 else (target_action - self._last_targets[0]) / denom
        else:
            vel_target = np.zeros_like(target_action)
        self._last_targets.append(target_action.copy())
        self._last_targets = self._last_targets[-2:]
        _, new_cmd = self._tracker.track(observed_arr, target_action, vel_target, interval_s)
        post_smooth_action = self._command_transform.apply(new_cmd)
        send_timestamp = time.time()
        self.apply_action(new_cmd)
        return interval_s, [
            {
                "timestamp": send_timestamp,
                "action": raw_pre_smooth_action.tolist(),
                "source": "raw_pre_smooth_action",
            },
            {
                "timestamp": send_timestamp,
                "action": target_raw.tolist(),
                "source": "pre_smooth_action",
            },
            {
                "timestamp": send_timestamp,
                "action": list(post_smooth_action),
                "source": "post_smooth_action",
            },
        ]

    def apply_action(self, action: np.ndarray) -> None:
        command = self._command_transform.apply(action)
        self._actuator.apply(np.asarray(command, dtype=np.float32).reshape(-1))

    def close(self) -> None:
        self._actuator.close()

@dataclass
class OnDeviceMpcExecutor(BaseExecutor):
    planner_dims: tuple[int, ...]
    enable_init_action: bool = False
    init_action: list[float] = field(default_factory=list)
    init_steps: int = 100
    init_sleep_s: float = 0.01
    control_dt_s: float = 0.02
    obs_image_delay_ms: float = 55.0
    state_delay_s: float = 0.05
    max_prefill_states: int = 15
    heartbeat_history_len: int = 200
    infer_fixed_dims: list[int] = field(default_factory=list)
    infer_fixed_values: list[float] = field(default_factory=list)
    command_fixed_dims: list[int] = field(default_factory=list)
    command_fixed_values: list[float] = field(default_factory=list)
    actuator: BaseActuator | None = None
    mpc_contact_threshold: float = 0.15
    mpc_contact_hold_min: int = 8
    mpc_tau: float = 0.15
    mpc_horizon_n: int = 15
    mpc_w_track0: float = 10.0
    mpc_track_decay: float = 1.0
    mpc_w_cmd: float = 0.0
    mpc_w_yx: float = 0.0
    mpc_w_dy: float = 0.0
    mpc_w_ddy: float = 10.0
    mpc_contact_v_scale: float = 0.4
    mpc_q_min: list[float] = field(default_factory=list)
    mpc_q_max: list[float] = field(default_factory=list)
    mpc_v_max: list[float] = field(default_factory=list)
    mpc_dy_max: list[float] = field(default_factory=list)
    mpc_ddy_max: list[float] = field(default_factory=list)
    gripper_heartbeat_lookahead_ms: float = 90.0
    gripper_lookahead_dims: tuple[int, ...] = (6, 13)
    emergency_pause_s: float = 0.150

    @classmethod
    def from_config(cls, cfg, actuator: BaseActuator):
        ex = cfg.executor
        return cls(
            planner_dims=tuple(ex.planner_dims),
            enable_init_action=bool(ex.enable_init_action),
            init_action=[float(v) for v in ex.init_action],
            init_steps=int(ex.init_steps),
            init_sleep_s=float(ex.init_sleep_s),
            control_dt_s=ex.control_dt_s,
            obs_image_delay_ms=ex.obs_image_delay_ms,
            state_delay_s=ex.state_delay_s,
            max_prefill_states=ex.max_prefill_states,
            heartbeat_history_len=ex.heartbeat_history_len,
            infer_fixed_dims=[int(v) for v in ex.infer_fixed_dims],
            infer_fixed_values=[float(v) for v in ex.infer_fixed_values],
            command_fixed_dims=[int(v) for v in ex.command_fixed_dims],
            command_fixed_values=[float(v) for v in ex.command_fixed_values],
            actuator=actuator,
            mpc_contact_threshold=ex.mpc_contact_threshold,
            mpc_contact_hold_min=ex.mpc_contact_hold_min,
            mpc_tau=ex.mpc_tau,
            mpc_horizon_n=ex.mpc_horizon_n,
            mpc_w_track0=ex.mpc_w_track0,
            mpc_track_decay=ex.mpc_track_decay,
            mpc_w_cmd=ex.mpc_w_cmd,
            mpc_w_yx=ex.mpc_w_yx,
            mpc_w_dy=ex.mpc_w_dy,
            mpc_w_ddy=ex.mpc_w_ddy,
            mpc_contact_v_scale=ex.mpc_contact_v_scale,
            mpc_q_min=list(ex.mpc_q_min),
            mpc_q_max=list(ex.mpc_q_max),
            mpc_v_max=list(ex.mpc_v_max),
            mpc_dy_max=list(ex.mpc_dy_max),
            mpc_ddy_max=list(ex.mpc_ddy_max),
            gripper_heartbeat_lookahead_ms=float(ex.gripper_heartbeat_lookahead_ms),
            gripper_lookahead_dims=tuple(int(v) for v in ex.gripper_lookahead_dims),
        )

    def __post_init__(self) -> None:
        self._actuator: BaseActuator = self.actuator
        self._planner_dims = tuple(int(d) for d in self.planner_dims)
        if not self._planner_dims:
            raise ValueError("executor.planner_dims must not be empty for ondevice_mpc")
        self.gripper_heartbeat_lookahead_ms = float(self.gripper_heartbeat_lookahead_ms)
        self.gripper_lookahead_dims = tuple(int(d) for d in self.gripper_lookahead_dims)
        self._infer_transform = _build_action_transform(
            self.infer_fixed_dims,
            self.infer_fixed_values,
            name="ondevice_mpc infer transform",
        )
        self._command_transform = _build_action_transform(
            self.command_fixed_dims,
            self.command_fixed_values,
            name="ondevice_mpc command transform",
        )
        nq = len(self._planner_dims)
        self._planner_cfg = MPCConfig(
            nq=nq,
            dt=float(self.control_dt_s),
            tau=float(self.mpc_tau),
            N=int(self.mpc_horizon_n),
            q_min=_expand_optional_array(self.mpc_q_min, nq, "mpc_q_min"),
            q_max=_expand_optional_array(self.mpc_q_max, nq, "mpc_q_max"),
            v_max=_expand_optional_array(self.mpc_v_max, nq, "mpc_v_max"),
            dy_max=_expand_optional_array(self.mpc_dy_max, nq, "mpc_dy_max"),
            ddy_max=_expand_optional_array(self.mpc_ddy_max, nq, "mpc_ddy_max"),
            w_track0=float(self.mpc_w_track0),
            track_decay=float(self.mpc_track_decay),
            w_cmd=float(self.mpc_w_cmd),
            w_yx=float(self.mpc_w_yx),
            w_dy=float(self.mpc_w_dy),
            w_ddy=float(self.mpc_w_ddy),
            contact_v_scale=float(self.mpc_contact_v_scale),
        )
        self._obs_image_delay_s = float(self.obs_image_delay_ms) / 1000.0
        self._state_delay_s = max(0.0, float(self.state_delay_s))
        self._max_prefill_states = int(self.max_prefill_states)

        try:
            self._planner = AcadosPlanner(self._planner_cfg)
        except Exception:
            self._planner = None
        self._estimator = ReplayEstimator(
            nq=len(self._planner_dims), tau=self._planner_cfg.tau
        )
        self._estimator.set_contact_params(
            threshold=float(self.mpc_contact_threshold), hold_min=int(self.mpc_contact_hold_min)
        )
        self._progress = ProgressManager()
        self._future_actions: list[list[float]] = []
        self._raw_future_actions: list[list[float]] = []
        self._action_queue_lock = threading.Lock()
        self._action_queue_revision = 0
        self._action_head_touch_counter = 0
        self._exec_step_count = 0
        self._last_executed_action: list[float] | None = None
        self._single_prefill_bootstrap_action: list[float] | None = None
        self._heartbeat_action_history: deque[dict] = deque(maxlen=max(1, int(self.heartbeat_history_len)))
        self._state_obs_history: deque[dict] = deque(maxlen=8000)
        self._state_obs_history_lock = threading.Lock()
        self._planner_runtime_lock = threading.Lock()
        self._emergency_lock = threading.Lock()
        self._emergency_generation = 0
        self._emergency_pause_until_mono = 0.0

    def normalize_infer_actions(self, action_list) -> list[list[float]]:
        if not isinstance(action_list, (list, tuple)):
            raise TypeError(f"invalid action_list type: {type(action_list)}")
        return [self._infer_transform.apply(action) for action in action_list]

    def apply_action(self, action: np.ndarray) -> None:
        command = self._command_transform.apply(action)
        self._actuator.apply(np.asarray(command, dtype=np.float32).reshape(-1))

    def close(self) -> None:
        self._actuator.close()

    def record_observation(self, action: list[float] | np.ndarray | None, timestamp: float) -> None:
        if action is None:
            return
        with self._state_obs_history_lock:
            self._state_obs_history.append({"timestamp": float(timestamp), "action": list(action)})

    def _get_delayed_state_snapshot(self) -> dict | None:
        target_time = time.time() - self._state_delay_s
        with self._state_obs_history_lock:
            if not self._state_obs_history:
                return None
            selected = self._state_obs_history[0]
            for item in reversed(self._state_obs_history):
                if float(item["timestamp"]) <= target_time:
                    selected = item
                    break
        return {"timestamp": float(selected["timestamp"]), "action": list(selected["action"])}

    def _mark_emergency_pause(self) -> int:
        with self._emergency_lock:
            self._emergency_generation += 1
            pause_until = time.monotonic() + float(self.emergency_pause_s)
            if pause_until > self._emergency_pause_until_mono:
                self._emergency_pause_until_mono = pause_until
            return int(self._emergency_generation)

    def sleep_if_emergency(self) -> bool:
        with self._emergency_lock:
            pause_until = float(self._emergency_pause_until_mono)
        now = time.monotonic()
        if now >= pause_until:
            return False
        time.sleep(max(0.0, pause_until - now))
        return True

    def _get_emergency_generation(self) -> int:
        with self._emergency_lock:
            return int(self._emergency_generation)

    def _emergency_generation_changed(self, generation: int) -> bool:
        with self._emergency_lock:
            return int(generation) != int(self._emergency_generation)

    def _predict_steps_with_history(
        self,
        latency_s: float | None,
        request_time: float | None,
        history: list[dict] | None,
        future_queue: list[list[float]] | None,
        control_dt_s: float,
    ) -> int:
        if latency_s is None:
            latency_s = 0.0
        control_dt_s = max(1e-6, float(control_dt_s))
        if request_time is None or not future_queue:
            return max(1, int(math.ceil(float(latency_s) / control_dt_s)))
        if history is None:
            history = []
        target_time = float(request_time) + max(0.0, float(latency_s))
        prev_action = history[-1]["action"] if history else None
        prev_time = float(history[-1]["timestamp"]) if history else float(request_time)
        future_times: list[float] = []
        for action in future_queue:
            if prev_action is None:
                cur_time = prev_time
            else:
                cur_time = prev_time + control_dt_s
            future_times.append(float(cur_time))
            prev_action = action
            prev_time = cur_time
        predicted_idx = None
        for idx, cur_time in enumerate(future_times):
            if cur_time >= target_time:
                predicted_idx = idx
                break
        if predicted_idx is None:
            extra = int(math.ceil((target_time - future_times[-1]) / control_dt_s))
            if extra < 0:
                extra = 0
            predicted_idx = len(future_times) - 1 + extra
        if predicted_idx < 1:
            predicted_idx = 1
        return int(predicted_idx)

    def on_infer_failure(self, _reason: str, current_state: list[float] | None = None) -> None:
        self._mark_emergency_pause()
        stop_action = list(current_state) if current_state is not None else (
            list(self._last_executed_action) if self._last_executed_action is not None else None
        )
        with self._action_queue_lock:
            self._future_actions = []
            self._raw_future_actions = []
            self._heartbeat_action_history.clear()
            self._progress.progress = 0.0
            self._last_executed_action = list(stop_action) if stop_action is not None else None
            self._single_prefill_bootstrap_action = None
            self._action_queue_revision += 1
            self._action_head_touch_counter += 1

        q0_plan = None
        if stop_action is not None:
            q0_plan = self._dims_from_action(stop_action)
        with self._planner_runtime_lock:
            self._estimator = ReplayEstimator(
                nq=len(self._planner_dims), tau=self._planner_cfg.tau
            )
            self._estimator.set_contact_params(
                threshold=float(self.mpc_contact_threshold), hold_min=int(self.mpc_contact_hold_min)
            )
            if self._planner is not None:
                if q0_plan is not None:
                    self._planner.reset(q0_plan)
                else:
                    self._planner.initialized = False
                    self._planner._contact_mode_cached = None
        with self._state_obs_history_lock:
            self._state_obs_history.clear()

    def prime_startup(
        self,
        anchor_action: list[float] | np.ndarray | None,
        *,
        anchor_timestamp: float | None = None,
        bootstrap_action: bool = True,
    ) -> None:
        anchor = None
        if anchor_action is not None:
            anchor = list(np.asarray(anchor_action, dtype=np.float32).reshape(-1))

        with self._action_queue_lock:
            self._future_actions = []
            self._raw_future_actions = []
            self._heartbeat_action_history.clear()
            self._progress.progress = 0.0
            self._action_queue_revision = 0
            self._action_head_touch_counter = 0
            self._exec_step_count = 0
            self._last_executed_action = list(anchor) if anchor is not None else None
            self._single_prefill_bootstrap_action = (
                list(anchor) if bootstrap_action and anchor is not None else None
            )

        with self._planner_runtime_lock:
            self._estimator = ReplayEstimator(
                nq=len(self._planner_dims), tau=self._planner_cfg.tau
            )
            self._estimator.set_contact_params(
                threshold=float(self.mpc_contact_threshold), hold_min=int(self.mpc_contact_hold_min)
            )

            if self._planner is not None:
                if anchor is not None:
                    self._planner.reset(self._dims_from_action(anchor))
                else:
                    self._planner.initialized = False
                    self._planner._contact_mode_cached = None

        with self._state_obs_history_lock:
            self._state_obs_history.clear()

    def prepare_infer_context(
        self,
        latency_s: float,
        current_state: list[float] | None = None,
        image_timestamp: float | None = None,
    ) -> tuple[dict, list[list[float]]]:
        request_time = time.time()
        with self._action_queue_lock:
            bootstrap_action = None
            if self._single_prefill_bootstrap_action is not None:
                bootstrap_action = list(self._single_prefill_bootstrap_action)
                if not self._future_actions:
                    self._future_actions.append(list(bootstrap_action))
                if self._last_executed_action is None:
                    self._last_executed_action = list(bootstrap_action)
            if bootstrap_action is not None:
                context = {
                    "exec_step_at_request": self._exec_step_count,
                    "predicted_idx": 0,
                    "predicted_steps": 0,
                    "padded": False,
                    "pad_action": None,
                    "request_time": request_time,
                    "single_prefill_bootstrap": True,
                }
                return context, [list(bootstrap_action)]

            if not self._future_actions and current_state is not None:
                self._future_actions.append(list(current_state))
                if self._last_executed_action is None:
                    self._last_executed_action = list(current_state)
            history_snapshot = [
                {"timestamp": float(h["timestamp"]), "action": list(h["action"])}
                for h in self._heartbeat_action_history
            ]
            exec_step = self._exec_step_count
            queue_snapshot = [list(a) for a in self._future_actions]
            predicted_steps = self._predict_steps_with_history(
                latency_s=latency_s,
                request_time=request_time,
                history=history_snapshot,
                future_queue=queue_snapshot,
                control_dt_s=self.get_control_dt_s(),
            )

            queue_len_before_pad = len(self._future_actions)
            pad_action = list(queue_snapshot[-1]) if queue_snapshot else (
                list(current_state) if current_state is not None else None
            )
            padded = predicted_steps >= queue_len_before_pad

            while len(self._future_actions) <= predicted_steps and pad_action is not None:
                self._future_actions.append(list(pad_action))
            queue_snapshot = [list(a) for a in self._future_actions]

            history_actions = [h["action"] for h in history_snapshot]
            timeline = history_actions + queue_snapshot
            history_len = len(history_actions)
            predicted_timeline_idx = history_len + predicted_steps
            if predicted_timeline_idx >= len(timeline) and pad_action is not None:
                while len(timeline) <= predicted_timeline_idx:
                    timeline.append(list(pad_action))

            anchor_time = None
            if image_timestamp is not None:
                anchor_time = float(image_timestamp) - self._obs_image_delay_s

            start_idx = 0
            if history_snapshot and anchor_time is not None:
                start_idx = _find_nearest_history_index(history_snapshot, anchor_time)
            anchor_timeline_idx = predicted_timeline_idx
            if self._max_prefill_states > 0:
                max_end = start_idx + self._max_prefill_states - 1
                if anchor_timeline_idx > max_end:
                    anchor_timeline_idx = max_end
            if anchor_timeline_idx < start_idx:
                anchor_timeline_idx = start_idx

            state_trajectory = timeline[start_idx : anchor_timeline_idx + 1]
            anchor_idx_in_future = anchor_timeline_idx - history_len
            if not state_trajectory and current_state is not None:
                state_trajectory = [list(current_state)]
                anchor_idx_in_future = 0

            context = {
                "exec_step_at_request": exec_step,
                "predicted_idx": anchor_idx_in_future,
                "predicted_steps": predicted_steps,
                "padded": padded,
                "pad_action": pad_action,
                "request_time": request_time,
                "single_prefill_bootstrap": False,
            }
            return context, state_trajectory

    def _updated_future_queue(
        self,
        queue: list[list[float]],
        normalized_actions: list[list[float]],
        context: dict,
        executed_since_request: int,
    ) -> tuple[list[list[float]], bool]:
        predicted_idx = int(context["predicted_idx"])
        old_queue_len = len(queue)
        head_overwritten = False
        anchor_idx_current = predicted_idx - executed_since_request

        if anchor_idx_current >= 0:
            if context.get("padded"):
                pad_action = context.get("pad_action")
                if pad_action is not None and queue:
                    tail_start = len(queue)
                    for idx in range(len(queue) - 1, -1, -1):
                        if queue[idx] == pad_action:
                            tail_start = idx
                        else:
                            break
                    if tail_start <= anchor_idx_current:
                        anchor_idx_current = tail_start
            overwrite_start_idx = anchor_idx_current + 1
            prefix = queue[:overwrite_start_idx]
            updated_queue = prefix + normalized_actions
            head_overwritten = overwrite_start_idx <= 0 and old_queue_len > 0
        else:
            executed_after_anchor = executed_since_request - (predicted_idx + 1)
            if executed_after_anchor < 0:
                executed_after_anchor = 0
            if executed_after_anchor >= len(normalized_actions):
                executed_after_anchor = len(normalized_actions) - 1
            updated_queue = normalized_actions[executed_after_anchor:]
            head_overwritten = old_queue_len > 0
        return updated_queue, head_overwritten

    def update_actions(
        self,
        action_list: list[list[float]],
        context: dict,
        raw_action_list: list[list[float]] | None = None,
    ) -> None:
        exec_step_at_request = int(context["exec_step_at_request"])
        normalized_actions = [list(a) for a in action_list]
        normalized_raw_actions = None
        if raw_action_list:
            normalized_raw_actions = [list(a) for a in raw_action_list]

        with self._action_queue_lock:
            executed_since_request = self._exec_step_count - exec_step_at_request
            if executed_since_request < 0:
                executed_since_request = 0

            self._future_actions, head_overwritten = self._updated_future_queue(
                self._future_actions,
                normalized_actions,
                context,
                executed_since_request,
            )
            if normalized_raw_actions:
                self._raw_future_actions, _ = self._updated_future_queue(
                    self._raw_future_actions,
                    normalized_raw_actions,
                    context,
                    executed_since_request,
                )
            else:
                self._raw_future_actions = []

            if normalized_actions:
                if context.get("single_prefill_bootstrap"):
                    self._single_prefill_bootstrap_action = None
                self._action_queue_revision += 1
                if head_overwritten:
                    self._action_head_touch_counter += 1

    def get_pending_count(self) -> int:
        with self._action_queue_lock:
            return len(self._future_actions)

    def get_control_dt_s(self) -> float:
        return max(1e-3, float(self.control_dt_s))

    def _dims_from_action(self, action: list[float]) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        return arr[np.asarray(self._planner_dims, dtype=np.int64)].copy()

    def _merge_dims(self, base_action: list[float], planned_dims: np.ndarray) -> list[float]:
        merged = list(base_action)
        for idx, dim in enumerate(self._planner_dims):
            if dim < len(merged) and idx < len(planned_dims):
                merged[dim] = float(planned_dims[idx])
        return merged

    def tick(self, observed_state: list[float]) -> list[dict] | None:
        if self.sleep_if_emergency():
            return None
        loop_start = time.time()
        loop_generation = self._get_emergency_generation()
        delayed_state = self._get_delayed_state_snapshot()
        observed_full = None
        observed_plan = None
        state_ts = None
        if delayed_state is not None:
            observed_full = list(delayed_state.get("action", []))
            state_ts = float(delayed_state.get("timestamp", loop_start))
            observed_plan = self._dims_from_action(observed_full)
            if observed_plan is not None:
                self._estimator.push_observation(state_ts, observed_plan)
                self._estimator.step_contact_decay()

        lookahead = max(1, int(getattr(self._planner_cfg, "N", 15)))
        gripper_lookahead_idx = _gripper_heartbeat_lookahead_index(
            self.get_control_dt_s() * 1000.0,
            self.gripper_heartbeat_lookahead_ms,
        )
        future_queue_len = max(lookahead, gripper_lookahead_idx + 1)
        with self._action_queue_lock:
            queue_revision = self._action_queue_revision
            queue_head_touch = self._action_head_touch_counter
            future = [list(a) for a in list(self._future_actions)[:future_queue_len]]
            raw_future = [list(a) for a in list(self._raw_future_actions)[:future_queue_len]]
        if not future:
            fallback = (
                list(self._last_executed_action)
                if self._last_executed_action is not None
                else list(observed_state)
            )
            future = [fallback]
        while len(future) < future_queue_len:
            future.append(list(future[-1]))
        while raw_future and len(raw_future) < future_queue_len:
            raw_future.append(list(raw_future[-1]))

        pre_mpc_action = _apply_gripper_heartbeat_lookahead(
            future[0],
            future,
            gripper_lookahead_idx,
            self.gripper_lookahead_dims,
        )
        if pre_mpc_action is None:
            pre_mpc_action = list(future[0])
        raw_pre_mpc_action = None
        if raw_future:
            raw_pre_mpc_action = _apply_gripper_heartbeat_lookahead(
                raw_future[0],
                raw_future,
                gripper_lookahead_idx,
                self.gripper_lookahead_dims,
            )
            if raw_pre_mpc_action is None:
                raw_pre_mpc_action = list(raw_future[0])
        target_action = list(pre_mpc_action)
        alpha = 1.0
        q_hat = None
        if observed_plan is not None:
            try:
                q_hat = self._estimator.estimate_now(loop_start)
            except Exception:
                q_hat = observed_plan
        if self._planner is not None and q_hat is not None:
            try:
                ai_future = np.asarray([self._dims_from_action(a) for a in future[:lookahead]], dtype=np.float32)
                with self._planner_runtime_lock:
                    out = self._planner.solve(
                        q_hat=q_hat,
                        ai_future=ai_future,
                        contact_mode=bool(self._estimator.contact_flag),
                    )
                target_action = self._merge_dims(target_action, out.y_cmd)
                alpha = float(out.alpha)
            except Exception:
                alpha = 0.0

        if self._emergency_generation_changed(loop_generation):
            self._progress.progress = 0.0
            return None

        with self._action_queue_lock:
            if (self._action_queue_revision != queue_revision
                    and self._action_head_touch_counter != queue_head_touch):
                self._progress.progress = 0.0
                return None

        if self._emergency_generation_changed(loop_generation):
            self._progress.progress = 0.0
            return None

        self._progress.advance(alpha)
        progress_reached_timestamp = None
        if self._progress.completed_steps() > 0:
            progress_reached_timestamp = time.time()
        send_timestamp = time.time()
        self.apply_action(np.asarray(target_action, dtype=np.float32))
        self._estimator.push_command(send_timestamp, self._dims_from_action(target_action))

        with self._action_queue_lock:
            if (self._action_queue_revision != queue_revision
                    and self._action_head_touch_counter != queue_head_touch):
                self._progress.progress = 0.0
            else:
                completed = self._progress.completed_steps()
                pop_count = min(completed, len(self._future_actions))
                reached_actions = [list(self._future_actions[i]) for i in range(pop_count)]
                for _ in range(pop_count):
                    self._future_actions.pop(0)
                raw_pop_count = min(pop_count, len(self._raw_future_actions))
                for _ in range(raw_pop_count):
                    self._raw_future_actions.pop(0)
                if pop_count > 0:
                    self._progress.consume_steps(pop_count)
                    reached_timestamp = (
                        progress_reached_timestamp
                        if progress_reached_timestamp is not None
                        else send_timestamp
                    )
                    for ra in reached_actions:
                        self._heartbeat_action_history.append(
                            {"timestamp": reached_timestamp, "action": ra}
                        )
                elif not self._future_actions and completed > 0:
                    self._progress.consume_steps(completed)

            self._last_executed_action = list(target_action)
            self._exec_step_count += 1

        records: list[dict] = []
        if raw_pre_mpc_action is not None:
            records.append(
                {
                    "timestamp": float(send_timestamp),
                    "action": list(raw_pre_mpc_action),
                    "source": "raw_pre_mpc_action",
                }
            )
        records.append(
            {"timestamp": float(send_timestamp), "action": list(pre_mpc_action), "source": "pre_mpc_action"}
        )
        records.append(
            {"timestamp": float(send_timestamp), "action": list(target_action), "source": "post_mpc_action"}
        )
        return records
