"""Import-safe RoboDojo adapter for single- and multi-arm tasks.

Action/observation foundation ported from OpenETA7/Stage2 at
ada4705e8baf740a4845544a0131c861e75e602d. Native outcome behavior corrected
against RoboDojo ee67a1468510da7624a089164402359f2afc72c8. No automatic
environment registration or change to the LIBERO path occurs on import.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np

from .outcome import reward_registration_present, snapshot
from .measurements import measure_effectors
from .native_reward import finite_scalar, step_with_native_reward
from .cameras import convert_camera
from .calibration import attach_calibrated_cameras
from .x5_calibration import attach_robot_calibration

class RoboDojoUnavailableError(RuntimeError):
    """RoboDojo runtime is unavailable in the current process."""


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _rotvec_to_quat(rotvec: Any) -> np.ndarray:
    value = np.asarray(rotvec, dtype=np.float32)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("delta_rotvec must be a finite 3-vector")
    angle = float(np.linalg.norm(value))
    if angle < 1e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    axis = value / angle
    half = angle * 0.5
    return np.r_[axis * np.sin(half), np.cos(half)].astype(np.float32)


def _compose_quat_xyzw(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    a, b, c, d = dq
    return np.array([w * a + x * d + y * c - z * b,
                     w * b - x * c + y * d + z * a,
                     w * c + x * b - y * a + z * d,
                     w * d - x * a - y * b - z * c], dtype=np.float32)


def _pose_wxyz_to_xyzw(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    return pose[[0, 1, 2, 4, 5, 6, 3]] if pose.size == 7 else pose


def _pose_xyzw_to_wxyz(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    return pose[[0, 1, 2, 6, 3, 4, 5]] if pose.size == 7 else pose


def _gripper_scalar(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape not in ((), (1,)):
        raise ValueError('Gripper openness requires one measured numeric value')
    result = float(raw.reshape(-1)[0])
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('Gripper openness must be finite in [0, 1]')
    return result


def _reported_arm_state(state, role):
    prefix = '' if role == 'primary' else role + '_'
    joints = state.get(prefix + 'arm_joint_state', state.get(prefix + 'joint_state'))
    joints = np.asarray(joints, dtype=np.float32)
    pose = _pose_wxyz_to_xyzw(state.get(prefix + 'ee_pose'))
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError(f'Missing or invalid six-joint observation for {role}')
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError(f'Missing or invalid reported pose for {role}')
    result = {'joint_positions': joints, 'ee_pose': pose}
    grip = state.get(prefix + 'gripper_open', state.get(prefix + 'ee_joint_state'))
    if grip is not None:
        # Pinned ObsManager derives ee_joint_state from prev_control, not
        # physical joint_pos. Keep the command for action preservation, but
        # never present it as measured closure or grasp feedback.
        result['gripper_command'] = _gripper_scalar(grip)
        result['gripper_state'] = {'commanded_openness': result['gripper_command'],
                                   'measurement_available': False}
    return result


class RoboDojoEnv(gym.Env):
    """One adapter whose arm capability is selected by ``arm_roles``."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 25}

    def __init__(self, task_name: str, *, seed: int = 0, robot: str = "x5",
                 robot_variant: str | None = None, arm_roles: tuple[str, ...] | list[str] | None = None,
                 render_mode: str | None = "rgb_array", backend_env: Any | None = None,
                 max_episode_steps: int = 1000, interactive: bool = False, **kwargs: Any) -> None:
        super().__init__()
        inferred = ("left", "right") if robot_variant == "dual_x5" else ("primary",)
        self._arm_roles = tuple(arm_roles or inferred)
        if self._arm_roles not in {('primary',), ('left', 'right')}:
            raise ValueError("The X5 backend supports primary or the ordered left/right pair")
        self._robot_variant = robot_variant or ("dual_x5" if len(self._arm_roles) > 1 else "single_x5")
        self._backend = "robodojo"
        self._task_name, self._seed, self._robot = task_name, int(seed), robot
        self.render_mode, self._env = render_mode, backend_env
        self._step_count, self._max_episode_steps = 0, int(max_episode_steps)
        self._interactive = bool(interactive)
        self._reward_registered = False
        self._terminal = False
        self._state_valid = False
        self._last_obs: dict[str, Any] = {}
        self.action_space = gym.spaces.Dict({})
        self.observation_space = gym.spaces.Dict({})

    @property
    def openeta_control_spec(self) -> dict[str, Any]:
        """Describe the structured control contract consumed by MCP tools."""
        return {
            "schema_version": "openeta.sim_control.v1",
            "controller": {
                "controller_id": "robodojo.eval_env_ik",
                "command_interface": "structured_ee_or_joint_position",
                "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
                "supports_position": True,
                "supports_orientation": True,
            },
            "cartesian_delta": {
                "supported": True,
                "command_frame": "world",
                "position_scale_m": 0.01,
                "rotation_scale_rad": 0.1,
                "arm_roles": list(self._arm_roles),
            },
            "gripper": {"supported": True, "range": [0.0, 1.0]},
        }

    def _ensure_backend(self) -> Any:
        if self._env is None:
            raise RoboDojoUnavailableError(
                f"RoboDojo task was not created by the isolated bench worker "
                f"(task={self._task_name!r}, root={os.environ.get('ROBODOJO_ROOT', '')!r})"
            )
        return self._env

    @staticmethod
    def _normalise_action(action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise TypeError("RoboDojo action must be a dictionary")
        result = dict(action)
        mode = str(result.get("mode", "ee_delta"))
        if mode not in {"ee_delta", "ee_pose", "joint_position"}:
            raise ValueError(f"unsupported RoboDojo action mode: {mode}")
        result["mode"] = mode
        for key in ("delta_xyz", "delta_rotvec", "positions", "pose", "ee_pose"):
            if key in result:
                arr = np.asarray(result[key], dtype=np.float32)
                if not np.isfinite(arr).all():
                    raise ValueError(f"non-finite values in action.{key}")
                result[key] = arr.tolist()
        if "gripper" in result:
            grip = float(np.asarray(result["gripper"]).reshape(-1)[0])
            if not np.isfinite(grip) or not 0.0 <= grip <= 1.0:
                raise ValueError("gripper must be finite in [0, 1]")
            result["gripper"] = grip
        return result

    @staticmethod
    def _arm_action(role: str, value: dict[str, Any], current: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        if not isinstance(value, dict):
            raise TypeError(f"arm {role!r} action must be a dictionary")
        mode = str(value.get("mode", "joint_position"))
        current = current or {}
        native: dict[str, Any]
        if mode == "ee_delta":
            xyz = np.asarray(value.get("delta_xyz", [0, 0, 0]), dtype=np.float32)
            rot = np.asarray(value.get("delta_rotvec", [0, 0, 0]), dtype=np.float32)
            pose = np.asarray(current.get("ee_pose", []), dtype=np.float32)
            if xyz.shape != (3,) or rot.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(rot).all():
                raise ValueError(f"{role}.ee_delta values must be finite 3-vectors")
            if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
                raise ValueError(f"{role}.ee_delta requires a valid current ee_pose")
            target = pose.copy()
            target[:3] += xyz
            # The MCP controller computes target * inverse(current): a world
            # rotation increment. Apply it on the left. Right multiplication
            # would interpret the same increment in the moving tool frame.
            target[3:] = _compose_quat_xyzw(
                _rotvec_to_quat(rot), target[3:] / np.linalg.norm(target[3:]))
            target[3:] /= np.linalg.norm(target[3:])
            key = "arm_ee_pose" if role == "primary" else f"{role}_ee_pose"
            native = {key: _pose_xyzw_to_wxyz(target).tolist()}
        elif mode == "ee_pose":
            target = np.array(value.get("pose", value.get("ee_pose", [])), dtype=np.float32, copy=True)
            if target.shape != (7,) or not np.isfinite(target).all() or np.linalg.norm(target[3:]) < 1e-8:
                raise ValueError(f"{role}.ee_pose must be a finite 7-vector with non-zero quaternion")
            target[3:] /= np.linalg.norm(target[3:])
            key = "arm_ee_pose" if role == "primary" else f"{role}_ee_pose"
            native = {key: _pose_xyzw_to_wxyz(target).tolist()}
        elif mode == "joint_position":
            positions = np.asarray(value.get("positions", current.get("joint_positions", [])), dtype=np.float32)
            if positions.size == 0:
                positions = np.asarray(current.get("joint_positions", []), dtype=np.float32)
            if positions.ndim != 1 or positions.size != 6 or not np.isfinite(positions).all():
                raise ValueError(f"{role}.positions must contain six finite values")
            key = "arm_joint_state" if role == "primary" else f"{role}_arm_joint_state"
            native = {key: positions.tolist()}
        else:
            raise ValueError(f"unsupported RoboDojo action mode: {mode}")
        if "gripper" in value:
            gripper = _gripper_scalar(value['gripper'])
            key = "ee_joint_state" if role == "primary" else f"{role}_ee_joint_state"
            native[key] = [gripper]
        return mode, native

    def _to_native_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise TypeError("RoboDojo action must be a dictionary")
        if "robodojo_action" in action:
            raise ValueError("Raw native action bypass is not part of the harness contract")
        if len(self._arm_roles) == 1:
            if "arms" in action:
                arms = action["arms"]
                if not isinstance(arms, dict) or set(arms) != set(self._arm_roles):
                    raise ValueError(f"single-arm action must contain only {self._arm_roles[0]!r}")
                value = arms[self._arm_roles[0]]
            else:
                if action.get("arm", "primary") not in {"primary", self._arm_roles[0]}:
                    raise ValueError("single-arm RoboDojo only supports arm='primary'")
                value = action
            current = self._last_obs.get("proprio", {})
            _, native = self._arm_action("primary", value, current)
            return native
        arms = action.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(self._arm_roles):
            raise ValueError(f"multi-arm action must contain exactly {self._arm_roles}")
        native: dict[str, Any] = {}
        modes: list[str] = []
        for role in self._arm_roles:
            current = self._last_obs.get("proprio", {}).get("arms", {}).get(role, {})
            mode, encoded = self._arm_action(role, arms[role], current)
            # RoboDojo's dual control manager resolves both gripper state keys on
            # every action, even when the caller targets only one side. Preserve
            # the unselected side's last commanded opening instead of sending an
            # incomplete native action.
            if role != "primary" and f"{role}_ee_joint_state" not in encoded:
                if 'gripper_command' not in current:
                    raise ValueError(f"Cannot preserve {role} gripper without its previous command")
                grip = _gripper_scalar(current['gripper_command'])
                encoded[f"{role}_ee_joint_state"] = [grip]
            modes.append(mode)
            native.update(encoded)
        if len(set(modes)) != 1:
            raise ValueError("RoboDojo multi-arm actions must use one mode for all arms")
        sync = action.get("synchronization", {})
        if sync and (sync.get("barrier", True) is not True or int(sync.get("max_skew_steps", 0)) != 0):
            raise ValueError("RoboDojo multi-arm actions require barrier=true and max_skew_steps=0")
        return native

    @staticmethod
    def _convert_obs(raw: Any, task_name: str = "fixture", robot: str = "x5", roles: tuple[str, ...] = ("primary",)) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise TypeError(f"RoboDojo observation must be dict, got {type(raw).__name__}")
        cameras = {}
        for name, frame in (raw.get("vision", {}) or {}).items():
            if not isinstance(frame, dict):
                continue
            cameras[str(name)] = convert_camera(frame)
        state = raw.get("state", {}) or {}
        if len(roles) == 1:
            proprio = _reported_arm_state(state, 'primary')
        else:
            arms = {role: _reported_arm_state(state, role) for role in roles}
            proprio = {"arms": arms, "joint_positions": np.concatenate([arms[r]["joint_positions"] for r in roles]) if arms else np.array([], dtype=np.float32)}
        return {"cameras": cameras, "proprio": proprio, "task_description": str(raw.get("instruction", task_name)), "objects": raw.get("objects", []) if isinstance(raw.get("objects", []), list) else [], "metadata": {"backend": "robodojo", "task": task_name, "robot": robot, "robot_variant": "dual_x5" if len(roles) > 1 else "single_x5", "schema_version": "robodojo-stage2-unified-v1"}}

    def _capture_observation(self, raw):
        from .observation_clock import capture_clock
        self._observation_clock = capture_clock(self._ensure_backend(), self._step_count)
        observation = self._convert_obs(raw, self._task_name, self._robot, self._arm_roles)
        attach_calibrated_cameras(self._ensure_backend(), observation)
        attach_robot_calibration(self._ensure_backend(), observation)
        measurements = measure_effectors(self._ensure_backend(), self._arm_roles)
        entries = (observation['proprio']['arms'] if len(self._arm_roles) > 1
                   else {'primary':observation['proprio']})
        for role,entry in entries.items():
            # Command and actual measurement remain distinct, even when a
            # held object prevents the fingers reaching the close command.
            grip = entry.setdefault('gripper_state', {})
            grip.update(measurements[role])
            if grip.get('openness') is not None and grip.get('commanded_openness') is not None:
                expected = grip['commanded_openness']
                mapping = grip.get('normalization')
                if mapping:
                    low, high = mapping['command_joint_limits']
                    p_low, p_high = mapping['physical_joint_limits']
                    fraction = expected if mapping['sign'] == 1 else 1-expected
                    requested = low + fraction*(high-low)
                    reachable = float(np.clip(requested, p_low, p_high))
                    expected = (reachable-p_low)/(p_high-p_low)
                    if mapping['sign'] == -1:
                        expected = 1-expected
                    grip.update(commanded_joint_target=requested,
                                reachable_joint_target=reachable,
                                command_exceeds_physical_limits=not p_low-1e-7 <= requested <= p_high+1e-7,
                                expected_reachable_openness=expected)
                grip['position_error_from_command'] = grip['openness']-expected
        return observation

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        from .episode_protocol import reset_protocol
        reset_protocol(self)
        self._checked_controller = None
        self._checked_marks = {}
        self._completion_trace = None
        if hasattr(self, "_completion_profile"):
            del self._completion_profile
        self._state_valid = False
        self._reward_registered = False
        if seed is not None:
            self._seed = int(seed)
        env = self._ensure_backend()
        from .physical_bindings import retain_rigid_bindings
        retain_rigid_bindings(env)
        from .initialization_profile import prepare_reset
        prepare_reset(self)
        result = env.reset(seed=[self._seed])
        retain_rigid_bindings(env)
        raw = result[0] if isinstance(result, tuple) else result
        if raw is None and callable(getattr(env, "get_obs", None)): raw = env.get_obs()
        self._step_count = 0
        self._terminal = False
        self._reward_registered = False
        self._last_obs = self._capture_observation(raw or {})
        # Reset is owned by this adapter. Register rewards exactly once after
        # native reset, matching run_eval; "interactive" never changes scoring.
        register_reward = getattr(env, "run_reward", None)
        if not callable(register_reward):
            raise RuntimeError('RoboDojo task has no official run_reward registration')
        register_reward()
        register_score = getattr(env, "get_score", None)
        if callable(register_score):
            register_score()
        self._reward_registered = reward_registration_present(env)
        if not self._reward_registered:
            raise RuntimeError('Official reward predicates are absent; refusing empty-manager success')
        from .support_phase import initialize_support_phase
        initialize_support_phase(env)
        official = snapshot(env, reward_registered=True, action_index=0)
        self._terminal = official['terminal']
        self._state_valid = True
        from .episode_recording import start_recording
        start_recording(self)
        return self._last_obs, {"task": self._task_name, "backend": "robodojo", "robot_variant": self._robot_variant,
                               'official_outcome': official}

    def step(self, action):
        if not self._reward_registered:
            raise RuntimeError('Reset and register official rewards before action')
        if self._terminal:
            raise RuntimeError('Official episode has ended; create/reset a new episode to continue')
        if not self._state_valid:
            raise RuntimeError('Previous action outcome is unknown; reset before sending another action')
        env = self._ensure_backend(); native = self._to_native_action(action)
        # Any exception after dispatch may follow physical execution. Do not
        # allow a retry to preserve another arm from a stale observation.
        self._state_valid = False
        trace = getattr(self, "_completion_trace", None)
        if trace is None:
            result, evaluated = step_with_native_reward(env, native)
        else:
            with trace.observe(env.reward_manager):
                result, evaluated = step_with_native_reward(env, native)
        values = list(result) if isinstance(result, tuple) else [result]; raw = values[0] if values else None
        reward = values[1] if len(values) > 1 else 0.0; terminated = values[2] if len(values) > 2 else False; truncated = values[3] if len(values) > 3 else False; info = values[4] if len(values) > 4 and isinstance(values[4], dict) else {}
        if len(values) <= 1:
            if 'reward' not in evaluated:
                raise RuntimeError('Native step returned no reward and performed no reward evaluation')
            reward = evaluated['reward']
        reward = finite_scalar(reward, 'step reward')
        if raw is None and callable(getattr(env, "get_obs", None)): raw = env.get_obs()
        self._step_count += 1; self._last_obs = self._capture_observation(raw or {})
        if trace is not None:
            trace.seal(self._step_count)
        official = snapshot(env, reward_registered=self._reward_registered, action_index=self._step_count)
        # Do not forward an upstream info.success or success-like alias which
        # could bypass the benchmark codec. Preserve other diagnostic fields.
        info = {k:v for k,v in info.items() if k not in {'success', 'task_success', 'benchmark_success',
                'environment_success', 'checker_success', 'official_outcome'}}
        info.update({'official_outcome': official,
                     'reward_source': 'robodojo_reward_manager' if len(values) <= 1 else 'robodojo',
                     'robot_variant': self._robot_variant})
        if 'score' in evaluated:
            info['score'] = evaluated['score']
        if official['benchmark_success'] is not None:
            info['benchmark_success'] = official['benchmark_success']
        episode_done = bool(terminated or official['terminal'])
        exhausted = bool(truncated or self._step_count >= self._max_episode_steps)
        self._terminal = episode_done or exhausted
        self._state_valid = True
        from .episode_recording import capture_recording
        capture_recording(self)
        return self._last_obs, reward, episode_done, exhausted, info

    def render(self):
        env = self._ensure_backend(); return env.render() if callable(getattr(env, "render", None)) else None

    def close(self):
        recorder = getattr(self, '_episode_recording', None)
        if recorder is not None:
            recorder.close()
        if self._env is not None and callable(getattr(self._env, "close", None)): self._env.close()
        self._env = None
