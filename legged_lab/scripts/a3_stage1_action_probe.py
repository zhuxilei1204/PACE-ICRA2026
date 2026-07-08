"""Probe constant A3 Stage-1 leg actions for drift and recovery control."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


@dataclass
class Candidate:
    name: str
    values: dict[str, float]
    joint_pos: dict[str, float] | None = None


def _candidate_specs(magnitudes: list[float]) -> list[Candidate]:
    candidates = [Candidate("zero", {})]
    groups = {
        "hip_pitch": [".*_hip_pitch_joint"],
        "knee": [".*_knee_joint"],
        "ankle_pitch": [".*_ankle_pitch_joint"],
        "hip_roll_widen": ["left_hip_roll_joint", "right_hip_roll_joint"],
        "ankle_roll_widen": ["left_ankle_roll_joint", "right_ankle_roll_joint"],
    }
    for mag in magnitudes:
        for sign in (-1.0, 1.0):
            val = sign * mag
            candidates.extend(
                [
                    Candidate(f"hip_pitch_{val:+.1f}", {".*_hip_pitch_joint": val}),
                    Candidate(f"knee_{val:+.1f}", {".*_knee_joint": val}),
                    Candidate(f"ankle_pitch_{val:+.1f}", {".*_ankle_pitch_joint": val}),
                    Candidate(
                        f"hip_knee_{val:+.1f}",
                        {".*_hip_pitch_joint": val, ".*_knee_joint": -val},
                    ),
                    Candidate(
                        f"hip_ankle_{val:+.1f}",
                        {".*_hip_pitch_joint": val, ".*_ankle_pitch_joint": -val},
                    ),
                    Candidate(
                        f"crouch_{val:+.1f}",
                        {".*_hip_pitch_joint": -val, ".*_knee_joint": val, ".*_ankle_pitch_joint": -0.5 * val},
                    ),
                ]
            )
            widen = {
                "left_hip_roll_joint": val,
                "right_hip_roll_joint": -val,
                "left_ankle_roll_joint": -0.5 * val,
                "right_ankle_roll_joint": 0.5 * val,
            }
            candidates.append(Candidate(f"widen_{val:+.1f}", widen))
            candidates.extend(
                [
                    Candidate(
                        f"hip_roll_same_{val:+.1f}",
                        {"left_hip_roll_joint": val, "right_hip_roll_joint": val},
                    ),
                    Candidate(
                        f"ankle_roll_same_{val:+.1f}",
                        {"left_ankle_roll_joint": val, "right_ankle_roll_joint": val},
                    ),
                    Candidate(
                        f"roll_lean_{val:+.1f}",
                        {
                            "left_hip_roll_joint": val,
                            "right_hip_roll_joint": val,
                            "left_ankle_roll_joint": -val,
                            "right_ankle_roll_joint": -val,
                        },
                    ),
                    Candidate(f"left_hip_roll_{val:+.1f}", {"left_hip_roll_joint": val}),
                    Candidate(f"right_hip_roll_{val:+.1f}", {"right_hip_roll_joint": val}),
                    Candidate(f"left_ankle_roll_{val:+.1f}", {"left_ankle_roll_joint": val}),
                    Candidate(f"right_ankle_roll_{val:+.1f}", {"right_ankle_roll_joint": val}),
                ]
            )
    return candidates


def _parse_magnitudes(value: str) -> list[float]:
    magnitudes = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not magnitudes:
        raise ValueError("--magnitudes must contain at least one value")
    return magnitudes


def _parse_joint_pos(value: str) -> dict[str, float]:
    if not value:
        return {}

    joint_pos: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --joint_pos item {item!r}; expected joint_name=value.")
        name, raw_value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --joint_pos item {item!r}; joint name is empty.")
        joint_pos[name] = float(raw_value.strip())
    return joint_pos


def _parse_joint_pos_candidates(value: str, repeat_each: int) -> list[Candidate]:
    if not value:
        return []

    candidates: list[Candidate] = []
    repeat_each = max(1, int(repeat_each))
    for raw_spec in value.split(";"):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        if ":" not in raw_spec:
            raise ValueError(
                f"Invalid --joint_pos_candidates item {raw_spec!r}; expected name:joint=value,joint=value."
            )
        name, raw_joint_pos = raw_spec.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --joint_pos_candidates item {raw_spec!r}; candidate name is empty.")
        joint_pos = _parse_joint_pos(raw_joint_pos)
        for idx in range(repeat_each):
            candidate_name = f"{name}_{idx:03d}" if repeat_each > 1 else name
            candidates.append(Candidate(candidate_name, {}, joint_pos.copy()))
    if not candidates:
        raise ValueError("--joint_pos_candidates did not contain any candidates.")
    return candidates


def _apply_joint_pos_override(env_cfg, joint_pos_override: dict[str, float]) -> None:
    if not joint_pos_override:
        return

    joint_pos = env_cfg.scene.robot.init_state.joint_pos.copy()
    joint_pos.update(joint_pos_override)
    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=env_cfg.scene.robot.init_state.replace(joint_pos=joint_pos)
    )

    default_override = getattr(env_cfg.robot, "default_joint_pos_override", None) or {}
    env_cfg.robot.default_joint_pos_override = {**default_override, **joint_pos_override}

    for event_name in ("reset_locomotion_joints", "reset_manipulation_joints"):
        reset_event = getattr(env_cfg.domain_rand.events, event_name, None)
        if reset_event is not None and "joint_pos" in reset_event.params:
            reset_event.params["joint_pos"] = {
                **reset_event.params["joint_pos"],
                **joint_pos_override,
            }


def _apply_per_env_joint_pos_candidates(
    env,
    candidates: list[Candidate],
    env_ids: torch.Tensor | None = None,
) -> None:
    if not any(candidate.joint_pos for candidate in candidates):
        return

    if env_ids is None:
        env_ids = torch.arange(len(candidates), dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    name_to_id = {name: idx for idx, name in enumerate(env.robot.joint_names)}
    for row_id, env_id in enumerate(env_ids.detach().cpu().tolist()):
        candidate = candidates[int(env_id)]
        for joint_name, value in (candidate.joint_pos or {}).items():
            if joint_name not in name_to_id:
                raise ValueError(f"Unknown joint in candidate {candidate.name!r}: {joint_name}")
            joint_pos[row_id, name_to_id[joint_name]] = float(value)

    joint_vel = torch.zeros_like(joint_pos)
    limits = env.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    env.robot.data.default_joint_pos[env_ids] = joint_pos
    env.robot.data.default_joint_vel[env_ids] = 0.0
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.robot.set_joint_position_target(joint_pos[:, env.action_joint_ids], env.action_joint_ids, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    if hasattr(env, "compute_perception"):
        env.compute_perception()
    if hasattr(env, "_sync_stage1_recovery_buffers"):
        env._sync_stage1_recovery_buffers()


def _scale_value(value, scale: float):
    if scale == 1.0:
        return value
    if isinstance(value, dict):
        return {key: float(val) * scale for key, val in value.items()}
    if isinstance(value, (float, int)):
        return float(value) * scale
    return value


def _apply_pd_scales(robot_cfg, args: argparse.Namespace) -> None:
    scales = {
        "waist": (args.waist_stiffness_scale, args.waist_damping_scale),
        "legs": (args.leg_stiffness_scale, args.leg_damping_scale),
        "feet": (args.feet_stiffness_scale, args.feet_damping_scale),
    }
    for group_name, (stiffness_scale, damping_scale) in scales.items():
        actuator = robot_cfg.actuators.get(group_name)
        if actuator is None:
            continue
        actuator.stiffness = _scale_value(actuator.stiffness, stiffness_scale)
        actuator.damping = _scale_value(actuator.damping, damping_scale)


def _apply_joint_preset(env_cfg, preset: str, root_z: float | None) -> None:
    if preset == "config" and root_z is None:
        return

    from legged_lab.assets.a3 import A3_PINGPONG_READY_JOINT_POS, A3_STABLE_STANDING_JOINT_POS

    joint_pos = env_cfg.scene.robot.init_state.joint_pos.copy()
    preset_joint_pos = {}
    if preset == "stable":
        preset_joint_pos = A3_STABLE_STANDING_JOINT_POS.copy()
    elif preset == "pingpong_ready":
        preset_joint_pos = A3_PINGPONG_READY_JOINT_POS.copy()
    elif preset != "config":
        raise ValueError(f"Unsupported joint preset: {preset}")
    joint_pos.update(preset_joint_pos)

    root_pos = tuple(env_cfg.scene.robot.init_state.pos)
    if root_z is not None:
        root_pos = (root_pos[0], root_pos[1], float(root_z))

    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=env_cfg.scene.robot.init_state.replace(pos=root_pos, joint_pos=joint_pos)
    )
    if preset_joint_pos and hasattr(env_cfg.robot, "default_joint_pos_override"):
        default_override = getattr(env_cfg.robot, "default_joint_pos_override", None) or {}
        env_cfg.robot.default_joint_pos_override = {**default_override, **preset_joint_pos}

    reset_event = getattr(env_cfg.domain_rand.events, "reset_locomotion_joints", None)
    if preset_joint_pos and reset_event is not None and "joint_pos" in reset_event.params:
        reset_event.params["joint_pos"] = {**reset_event.params["joint_pos"], **preset_joint_pos}


def _configure_env(env_cfg, args: argparse.Namespace, num_envs: int) -> None:
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.scene.max_episode_length_s = float(args.max_episode_length_s)
    env_cfg.scene.seed = int(args.seed)

    env_cfg.noise.add_noise = False
    for attr in (
        "lin_vel",
        "ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "height_scan",
        "ball_pos",
        "ball_linvel",
        "robot_pos",
        "perception",
        "ball_state",
    ):
        if hasattr(env_cfg.noise.noise_scales, attr):
            setattr(env_cfg.noise.noise_scales, attr, 0.0)

    env_cfg.domain_rand.events.push_robot = None
    if env_cfg.domain_rand.events.add_base_mass is not None:
        env_cfg.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    if env_cfg.domain_rand.events.physics_material is not None:
        env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (args.friction, args.friction)
        env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (args.friction, args.friction)
        env_cfg.domain_rand.events.physics_material.params["restitution_range"] = (0.0, 0.0)

    env_cfg.domain_rand.perception_delay.enable = False
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = {
        "x": (args.base_x, args.base_x),
        "y": (args.base_y, args.base_y),
        "yaw": (args.base_yaw, args.base_yaw),
    }
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_locomotion = getattr(env_cfg.domain_rand.events, "reset_locomotion_joints", None)
    if reset_locomotion is not None:
        if "position_range" in reset_locomotion.params:
            reset_locomotion.params["position_range"] = (1.0, 1.0)
        if "velocity_range" in reset_locomotion.params:
            reset_locomotion.params["velocity_range"] = (0.0, 0.0)

    reset_manipulation = getattr(env_cfg.domain_rand.events, "reset_manipulation_joints", None)
    if reset_manipulation is not None:
        if "position_range" in reset_manipulation.params:
            reset_manipulation.params["position_range"] = (0.0, 0.0)
        if "velocity_range" in reset_manipulation.params:
            reset_manipulation.params["velocity_range"] = (0.0, 0.0)

    env_cfg.ball.ball_speed_x_range = (0.0, 0.0)
    env_cfg.ball.ball_speed_y_range = (0.0, 0.0)
    env_cfg.ball.ball_speed_z_range = (0.0, 0.0)
    env_cfg.ball.ball_pos_y_range = (0.0, 0.0)
    env_cfg.ball.ball_max_eposide_length = 999999999.0
    env_cfg.ball.ball_reset_repeat = 1
    env_cfg.ball.max_serve_per_episode = 1_000_000

    env_cfg.robot.termination_robot_x_range = (-10.0, 10.0)
    env_cfg.robot.termination_robot_y_range = (-10.0, 10.0)
    if args.termination_min_z is not None:
        env_cfg.robot.termination_min_base_z = float(args.termination_min_z)
    _apply_pd_scales(env_cfg.scene.robot, args)
    _apply_joint_preset(env_cfg, args.joint_preset, args.root_z)


def _quat_wxyz_from_pitch(env, pitch: float) -> torch.Tensor:
    quat = torch.zeros((env.num_envs, 4), dtype=env.robot.data.root_quat_w.dtype, device=env.device)
    half_pitch = 0.5 * float(pitch)
    quat[:, 0] = math.cos(half_pitch)
    quat[:, 2] = math.sin(half_pitch)
    return quat


def _apply_recovery_state(env, args: argparse.Namespace) -> None:
    root_z = args.recovery_root_z
    pitch = float(args.recovery_pitch)
    pitch_rate = float(args.recovery_pitch_rate)
    if root_z is None and pitch == 0.0 and pitch_rate == 0.0:
        return

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    root_pose = torch.cat([env.robot.data.root_pos_w, env.robot.data.root_quat_w], dim=-1).detach().clone()
    if root_z is not None:
        root_pose[:, 2] = env.scene.env_origins[:, 2] + float(root_z)
    if pitch != 0.0:
        root_pose[:, 3:7] = _quat_wxyz_from_pitch(env, pitch)

    root_velocity = torch.zeros((env.num_envs, 6), dtype=root_pose.dtype, device=env.device)
    root_velocity[:, 4] = pitch_rate
    env.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    if hasattr(env, "compute_perception"):
        env.compute_perception()
    if hasattr(env, "_sync_stage1_recovery_buffers"):
        env._sync_stage1_recovery_buffers()


def _fill_action(env, candidates: list[Candidate]) -> torch.Tensor:
    actions = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float, device=env.device)
    for env_id, candidate in enumerate(candidates):
        for pattern, value in candidate.values.items():
            joint_ids, joint_names = env.robot.find_joints(pattern, preserve_order=True)
            action_indices = [env.action_joint_names.index(name) for name in joint_names if name in env.action_joint_names]
            for action_idx in action_indices:
                actions[env_id, action_idx] = float(value)
    return actions


def _robot_pos(env) -> torch.Tensor:
    if hasattr(env, "robot") and hasattr(env, "table"):
        return env.robot.data.root_link_pos_w - env.table.data.root_link_pos_w
    if hasattr(env, "robot_pos"):
        return env.robot_pos
    return env.robot.data.root_link_pos_w - env.scene.env_origins


def _flat_orientation_l2(env) -> torch.Tensor:
    if hasattr(env, "flat_orientation_l2_buf"):
        return env.flat_orientation_l2_buf
    gravity_xy = env.robot.data.projected_gravity_b[:, :2]
    return torch.sum(torch.square(gravity_xy), dim=1)


def _projected_gravity_xy(env) -> torch.Tensor:
    return env.robot.data.projected_gravity_b[:, :2]


def _root_roll_pitch_yaw(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quat = env.robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _reset_cause(env, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return value


def _actuator_group_joint_ids(env) -> dict[str, torch.Tensor]:
    groups = {
        "waist": ["waist_"],
        "hip": ["hip_"],
        "knee": ["knee_joint"],
        "ankle_pitch": ["ankle_pitch_joint"],
        "ankle_roll": ["ankle_roll_joint"],
    }
    result: dict[str, torch.Tensor] = {}
    for group_name, patterns in groups.items():
        ids = [
            joint_id
            for joint_id, joint_name in enumerate(env.robot.joint_names)
            if any(pattern in joint_name for pattern in patterns)
        ]
        if ids:
            result[group_name] = torch.tensor(ids, dtype=torch.long, device=env.device)
    return result


def _a3_reference_effort_limits(env) -> torch.Tensor:
    limits = torch.full_like(env.robot.data.applied_torque.float(), 1.0e9)
    for joint_id, joint_name in enumerate(env.robot.joint_names):
        if "waist_yaw_joint" in joint_name:
            limit = 220.0
        elif "waist_roll_joint" in joint_name:
            limit = 46.0
        elif "waist_pitch_joint" in joint_name:
            limit = 115.0
        elif "hip_" in joint_name:
            limit = 220.0
        elif "knee_joint" in joint_name:
            limit = 320.0
        elif "ankle_pitch_joint" in joint_name:
            limit = 118.2
        elif "ankle_roll_joint" in joint_name:
            limit = 54.75
        else:
            continue
        limits[:, joint_id] = limit
    return limits


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe A3 Stage-1 constant action drift.")
    parser.add_argument("--task", type=str, default="a3_tt_stage1_balance_move")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--magnitudes", type=str, default="1.0,2.0,3.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=4.0)
    parser.add_argument("--max_episode_length_s", type=float, default=999999999.0)
    parser.add_argument("--base_x", type=float, default=0.16)
    parser.add_argument("--base_y", type=float, default=0.35)
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--zero_only", action="store_true")
    parser.add_argument("--repeat_zero_envs", type=int, default=0)
    parser.add_argument(
        "--focus_candidates",
        type=str,
        default="",
        help="Comma-separated candidate names to repeat, e.g. zero,hip_pitch_+0.5.",
    )
    parser.add_argument("--repeat_each", type=int, default=1, help="Repeat each focused candidate across N envs.")
    parser.add_argument("--joint_preset", choices=["config", "stable", "pingpong_ready"], default="config")
    parser.add_argument("--root_z", type=float, default=None)
    parser.add_argument("--termination_min_z", type=float, default=None)
    parser.add_argument("--random_std", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument(
        "--joint_pos",
        type=str,
        default="",
        help="Comma-separated joint overrides, e.g. left_knee_joint=0.48,right_knee_joint=0.48.",
    )
    parser.add_argument(
        "--joint_pos_candidates",
        type=str,
        default="",
        help="Semicolon-separated per-candidate joint overrides, e.g. c1:left_knee_joint=0.46; c2:left_knee_joint=0.44.",
    )
    parser.add_argument("--recovery_root_z", type=float, default=None)
    parser.add_argument("--recovery_pitch", type=float, default=0.0)
    parser.add_argument("--recovery_pitch_rate", type=float, default=0.0)
    parser.add_argument("--waist_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--waist_damping_scale", type=float, default=1.0)
    parser.add_argument("--leg_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--leg_damping_scale", type=float, default=1.0)
    parser.add_argument("--feet_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--feet_damping_scale", type=float, default=1.0)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--summary_only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    joint_pos_candidates = _parse_joint_pos_candidates(args.joint_pos_candidates, args.repeat_each)
    if joint_pos_candidates:
        candidates = joint_pos_candidates
    elif args.zero_only:
        repeat_zero_envs = max(1, int(args.repeat_zero_envs))
        candidates = [Candidate(f"zero_{i:03d}", {}) for i in range(repeat_zero_envs)]
    else:
        candidates = _candidate_specs(_parse_magnitudes(args.magnitudes))
        if args.focus_candidates:
            by_name = {candidate.name: candidate for candidate in candidates}
            focus_names = [item.strip() for item in args.focus_candidates.split(",") if item.strip()]
            if not focus_names:
                raise ValueError("--focus_candidates was set but did not contain any candidate names.")
            repeat_each = max(1, int(args.repeat_each))
            repeated_candidates: list[Candidate] = []
            for name in focus_names:
                if name not in by_name:
                    valid = ", ".join(sorted(by_name))
                    raise ValueError(f"Unknown focus candidate {name!r}. Valid candidates: {valid}")
                for idx in range(repeat_each):
                    candidate_name = f"{name}_{idx:03d}" if repeat_each > 1 else name
                    repeated_candidates.append(Candidate(candidate_name, by_name[name].values.copy()))
            candidates = repeated_candidates

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, _ = task_registry.get_cfgs(args.task)
    joint_pos_override = _parse_joint_pos(args.joint_pos)
    _configure_env(env_cfg, args, len(candidates))
    _apply_joint_pos_override(env_cfg, joint_pos_override)
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, headless=bool(args.headless))

    try:
        _apply_per_env_joint_pos_candidates(env, candidates)
        _apply_recovery_state(env, args)
        actions = _fill_action(env, candidates)
        random_generator = None
        random_std = float(args.random_std)
        if random_std > 0.0:
            random_generator = torch.Generator(device=env.device)
            random_generator.manual_seed(int(args.random_seed))
        start_pos = _robot_pos(env).detach().clone()
        min_pos = start_pos.clone()
        max_pos = start_pos.clone()
        start_flat_l2 = _flat_orientation_l2(env).detach().clone()
        max_flat_l2 = start_flat_l2.clone()
        final_flat_l2 = start_flat_l2.clone()
        start_roll, start_pitch, _ = _root_roll_pitch_yaw(env)
        start_roll = start_roll.detach().clone()
        start_pitch = start_pitch.detach().clone()
        start_gravity_xy = _projected_gravity_xy(env).detach().clone()
        max_abs_roll = torch.abs(start_roll)
        max_abs_pitch = torch.abs(start_pitch)
        max_abs_gravity_xy = torch.abs(start_gravity_xy)
        final_roll = start_roll.clone()
        final_pitch = start_pitch.clone()
        final_gravity_xy = start_gravity_xy.clone()
        reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        low_z_reset_seen = torch.zeros_like(reset_seen)
        flat_reset_seen = torch.zeros_like(reset_seen)
        bad_posture_reset_seen = torch.zeros_like(reset_seen)
        timeout_seen = torch.zeros_like(reset_seen)
        first_reset = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        alive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        action_l2_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_max_abs = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        actuator_group_ids = _actuator_group_joint_ids(env)
        torque_abs_sum = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            for name in actuator_group_ids
        }
        torque_abs_max = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            for name in actuator_group_ids
        }
        effort_frac_sum = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            for name in actuator_group_ids
        }
        effort_frac_max = {
            name: torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            for name in actuator_group_ids
        }
        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                before_flat_l2 = _flat_orientation_l2(env).detach()
                max_flat_l2 = torch.maximum(max_flat_l2, before_flat_l2)
                step_actions = actions
                if random_generator is not None:
                    step_actions = actions + random_std * torch.randn(
                        actions.shape,
                        dtype=actions.dtype,
                        device=actions.device,
                        generator=random_generator,
                    )
                action_l2_sum += torch.sum(torch.square(step_actions), dim=1)
                action_max_abs = torch.maximum(action_max_abs, torch.max(torch.abs(step_actions), dim=1).values)
                _, _, reset_buf, _ = env.step(step_actions)
                if hasattr(env.robot.data, "applied_torque"):
                    abs_torque_all = torch.abs(env.robot.data.applied_torque.float())
                    effort_limits_all = _a3_reference_effort_limits(env).clamp_min(1e-6)
                    for group_name, joint_ids in actuator_group_ids.items():
                        abs_torque = abs_torque_all[:, joint_ids]
                        effort_fraction = abs_torque / effort_limits_all[:, joint_ids]
                        torque_abs_sum[group_name] += abs_torque.mean(dim=1)
                        torque_abs_max[group_name] = torch.maximum(torque_abs_max[group_name], abs_torque.max(dim=1).values)
                        effort_frac_sum[group_name] += effort_fraction.mean(dim=1)
                        effort_frac_max[group_name] = torch.maximum(
                            effort_frac_max[group_name], effort_fraction.max(dim=1).values
                        )
                if bool(reset_buf.any()) and any(candidate.joint_pos for candidate in candidates):
                    reset_ids = reset_buf.nonzero(as_tuple=False).flatten()
                    _apply_per_env_joint_pos_candidates(env, candidates, reset_ids)
                pos = _robot_pos(env).detach()
                step_flat_l2 = _flat_orientation_l2(env).detach()
                step_roll, step_pitch, _ = _root_roll_pitch_yaw(env)
                step_roll = step_roll.detach()
                step_pitch = step_pitch.detach()
                step_gravity_xy = _projected_gravity_xy(env).detach()
                max_flat_l2 = torch.maximum(max_flat_l2, step_flat_l2)
                final_flat_l2 = step_flat_l2
                max_abs_roll = torch.maximum(max_abs_roll, torch.abs(step_roll))
                max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(step_pitch))
                max_abs_gravity_xy = torch.maximum(max_abs_gravity_xy, torch.abs(step_gravity_xy))
                final_roll = step_roll
                final_pitch = step_pitch
                final_gravity_xy = step_gravity_xy
                min_pos = torch.minimum(min_pos, pos)
                max_pos = torch.maximum(max_pos, pos)
                low_z_reset_seen |= _reset_cause(env, "reset_low_z_buf")
                flat_reset_seen |= _reset_cause(env, "reset_flat_orientation_buf")
                bad_posture_reset_seen |= _reset_cause(env, "reset_bad_posture_duration_buf")
                timeout_seen |= _reset_cause(env, "reset_episode_timeout_buf")
                first_reset = torch.where(
                    reset_buf & (first_reset < 0),
                    torch.full_like(first_reset, step),
                    first_reset,
                )
                reset_seen |= reset_buf
                alive_steps += (~reset_buf).long()
        final_pos = _robot_pos(env).detach()
        dt = float(env.step_dt) * float(args.steps)

        rows = []
        if not args.summary_only:
            print("candidate                         dx      dy      dz     vx     reset  rst  x_range")
        for i, candidate in enumerate(candidates):
            dx = float(final_pos[i, 0] - start_pos[i, 0])
            dy = float(final_pos[i, 1] - start_pos[i, 1])
            dz = float(final_pos[i, 2] - start_pos[i, 2])
            vx = dx / dt
            row = {
                "candidate": candidate.name,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "vx": vx,
                "reset_seen": bool(reset_seen[i].detach().cpu()),
                "first_reset": int(first_reset[i].detach().cpu()),
                "alive_steps": int(alive_steps[i].detach().cpu()),
                "min_x": float(min_pos[i, 0]),
                "max_x": float(max_pos[i, 0]),
                "min_y": float(min_pos[i, 1]),
                "max_y": float(max_pos[i, 1]),
                "min_z": float(min_pos[i, 2]),
                "max_z": float(max_pos[i, 2]),
                "final_z": float(final_pos[i, 2]),
                "start_flat_l2": float(start_flat_l2[i]),
                "max_flat_l2": float(max_flat_l2[i]),
                "final_flat_l2": float(final_flat_l2[i]),
                "start_roll": float(start_roll[i]),
                "start_pitch": float(start_pitch[i]),
                "final_roll": float(final_roll[i]),
                "final_pitch": float(final_pitch[i]),
                "final_gravity_x": float(final_gravity_xy[i, 0]),
                "final_gravity_y": float(final_gravity_xy[i, 1]),
                "max_abs_roll": float(max_abs_roll[i]),
                "max_abs_pitch": float(max_abs_pitch[i]),
                "max_abs_gravity_x": float(max_abs_gravity_xy[i, 0]),
                "max_abs_gravity_y": float(max_abs_gravity_xy[i, 1]),
                "mean_action_l2": float(action_l2_sum[i] / max(args.steps, 1)),
                "max_action_abs": float(action_max_abs[i]),
                "low_z_reset_seen": bool(low_z_reset_seen[i].detach().cpu()),
                "flat_reset_seen": bool(flat_reset_seen[i].detach().cpu()),
                "bad_posture_reset_seen": bool(bad_posture_reset_seen[i].detach().cpu()),
                "timeout_seen": bool(timeout_seen[i].detach().cpu()),
            }
            for group_name in actuator_group_ids:
                row[f"{group_name}_torque_abs_mean"] = float(torque_abs_sum[group_name][i] / max(args.steps, 1))
                row[f"{group_name}_torque_abs_max"] = float(torque_abs_max[group_name][i])
                row[f"{group_name}_effort_frac_mean"] = float(effort_frac_sum[group_name][i] / max(args.steps, 1))
                row[f"{group_name}_effort_frac_max"] = float(effort_frac_max[group_name][i])
            rows.append(row)
            if not args.summary_only:
                print(
                    f"{candidate.name:<32} {dx:+.4f} {dy:+.4f} {dz:+.4f} {vx:+.4f} "
                    f"{int(row['reset_seen']):>5} {row['first_reset']:>4} "
                    f"{row['min_x']:+.3f}..{row['max_x']:+.3f} "
                    f"flat_max={row['max_flat_l2']:.4f} "
                    f"pitch_final={row['final_pitch']:+.4f}"
                )

        output_csv = args.output_csv
        if output_csv:
            path = Path(output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\n[A3 Stage1 Action Probe] CSV written to: {path}")
        if rows:
            print(
                "\n[A3 Stage1 Action Probe] metrics: "
                f"final_z_mean={statistics.fmean(row['final_z'] for row in rows):.4f}, "
                f"min_z={min(row['min_z'] for row in rows):.4f}, "
                f"max_flat_l2={max(row['max_flat_l2'] for row in rows):.4f}, "
                f"final_pitch_mean={statistics.fmean(row['final_pitch'] for row in rows):+.4f}, "
                f"max_abs_pitch={max(row['max_abs_pitch'] for row in rows):.4f}, "
                f"mean_dz={statistics.fmean(row['dz'] for row in rows):+.4f}, "
                f"mean_action_l2={statistics.fmean(row['mean_action_l2'] for row in rows):.6f}"
            )
        print(
            "\n[A3 Stage1 Action Probe] reset summary: "
            f"any={int(reset_seen.sum().detach().cpu())}/{env.num_envs}, "
            f"low_z={int(low_z_reset_seen.sum().detach().cpu())}, "
            f"flat={int(flat_reset_seen.sum().detach().cpu())}, "
            f"bad_posture={int(bad_posture_reset_seen.sum().detach().cpu())}, "
            f"timeout={int(timeout_seen.sum().detach().cpu())}"
        )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
