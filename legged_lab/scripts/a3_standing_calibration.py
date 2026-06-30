"""Batch-test A3 table-tennis ready stances.

This script is A3-only and does not modify training configuration. It creates
image-inspired crouched table-tennis stance candidates, holds each candidate
with position targets, and ranks them by free-base standing stability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

UPPER_BODY_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

TORSO_HEAD_JOINTS = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
)

LEFT_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

READY_ARM_JOINTS = {
    "left_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.55,
    "left_shoulder_yaw_joint": -0.25,
    "left_elbow_joint": 0.75,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.10,
    "right_shoulder_roll_joint": -0.55,
    "right_shoulder_yaw_joint": 0.35,
    "right_elbow_joint": 0.85,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": -0.20,
    "right_wrist_yaw_joint": 0.10,
}

ZERO_ARM_JOINTS = {
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

NEUTRAL_UPPER_BODY_JOINTS = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}


@dataclass(frozen=True)
class Candidate:
    index: int
    name: str
    root_z: float
    joint_pos: dict[str, float]


@dataclass
class Result:
    candidate: Candidate
    score: float
    survival_steps: int
    survived: bool
    reset_seen: bool
    max_abs_roll_pitch: float
    max_root_z_drift: float
    max_foot_slip: float
    both_feet_contact_ratio: float
    min_paddle_future_dist: float
    final_root_z: float
    final_roll: float
    final_pitch: float
    min_pitch: float
    max_pitch: float


def _parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float.")
    return [float(item) for item in items]


def _parse_vec3(value: str) -> list[float]:
    items = _parse_float_list(value)
    if len(items) != 3:
        raise ValueError("Expected three comma-separated floats.")
    return items


def _parse_str_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one item.")
    return items


def _load_measured_pose_json(path: str) -> dict[str, float] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    layout = payload.get("layout")
    q = payload.get("q")
    if not isinstance(layout, list) or not isinstance(q, list):
        raise ValueError("--measured_pose_json must contain list fields 'layout' and 'q'.")
    if len(layout) != len(q):
        raise ValueError(
            f"--measured_pose_json layout/q length mismatch: layout={len(layout)}, q={len(q)}."
        )
    pose: dict[str, float] = {}
    for joint_name, value in zip(layout, q):
        if not isinstance(joint_name, str):
            raise ValueError("--measured_pose_json layout entries must be joint name strings.")
        pose[joint_name] = float(value)
    return pose


def _measured_upper_pose(measured_pose: dict[str, float]) -> dict[str, float]:
    return {name: measured_pose[name] for name in UPPER_BODY_JOINTS if name in measured_pose}


def _measured_subset_pose(measured_pose: dict[str, float], joint_names: tuple[str, ...]) -> dict[str, float]:
    return {name: measured_pose[name] for name in joint_names if name in measured_pose}


def _parse_blend_alpha(arm_mode: str, prefix: str) -> float:
    if not arm_mode.startswith(prefix):
        raise ValueError(f"Expected arm mode starting with {prefix!r}, got {arm_mode!r}.")
    suffix = arm_mode.removeprefix(prefix)
    if suffix.startswith("_"):
        suffix = suffix[1:]
    try:
        alpha = float(suffix)
    except ValueError as exc:
        raise ValueError(f"Invalid blend alpha in arm mode {arm_mode!r}.") from exc
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"Blend alpha must be in [0, 1], got {alpha}.")
    return alpha


def _euler_xyz_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _base_pose(arm_mode: str, measured_pose: dict[str, float] | None = None) -> dict[str, float]:
    pose: dict[str, float] = {}
    pose.update(NEUTRAL_UPPER_BODY_JOINTS)
    if arm_mode == "ready":
        pose.update(READY_ARM_JOINTS)
    elif arm_mode == "zero":
        pose.update(ZERO_ARM_JOINTS)
    elif arm_mode == "default":
        pass
    elif arm_mode == "measured":
        if measured_pose is None:
            raise ValueError("crouch arm mode 'measured' requires --measured_pose_json.")
        pose.update(_measured_upper_pose(measured_pose))
    elif arm_mode == "measured_torso":
        if measured_pose is None:
            raise ValueError("crouch arm mode 'measured_torso' requires --measured_pose_json.")
        pose.update(ZERO_ARM_JOINTS)
        pose.update(_measured_subset_pose(measured_pose, TORSO_HEAD_JOINTS))
    elif arm_mode == "measured_arms":
        if measured_pose is None:
            raise ValueError("crouch arm mode 'measured_arms' requires --measured_pose_json.")
        pose.update(ZERO_ARM_JOINTS)
        pose.update(_measured_subset_pose(measured_pose, LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS))
    elif arm_mode == "measured_right_arm":
        if measured_pose is None:
            raise ValueError("crouch arm mode 'measured_right_arm' requires --measured_pose_json.")
        pose.update(ZERO_ARM_JOINTS)
        pose.update(_measured_subset_pose(measured_pose, RIGHT_ARM_JOINTS))
    elif arm_mode.startswith("measured_right_arm_blend_"):
        if measured_pose is None:
            raise ValueError("crouch arm mode 'measured_right_arm_blend_*' requires --measured_pose_json.")
        alpha = _parse_blend_alpha(arm_mode, "measured_right_arm_blend")
        pose.update(ZERO_ARM_JOINTS)
        measured_right_arm = _measured_subset_pose(measured_pose, RIGHT_ARM_JOINTS)
        for joint_name, measured_value in measured_right_arm.items():
            pose[joint_name] = alpha * measured_value
    else:
        raise ValueError(f"Unsupported arm mode: {arm_mode}")
    return pose


def _leg_pose(
    hip_pitch: float,
    knee: float,
    ankle_pitch: float,
    left_hip_roll: float,
    right_hip_roll: float,
    toe_out: float,
) -> dict[str, float]:
    ankle_roll_scale = 0.45
    return {
        "left_hip_pitch_joint": hip_pitch,
        "left_hip_roll_joint": left_hip_roll,
        "left_hip_yaw_joint": toe_out,
        "left_knee_joint": knee,
        "left_ankle_pitch_joint": ankle_pitch,
        "left_ankle_roll_joint": -ankle_roll_scale * left_hip_roll,
        "right_hip_pitch_joint": hip_pitch,
        "right_hip_roll_joint": right_hip_roll,
        "right_hip_yaw_joint": -toe_out,
        "right_knee_joint": knee,
        "right_ankle_pitch_joint": ankle_pitch,
        "right_ankle_roll_joint": -ankle_roll_scale * right_hip_roll,
    }


def _generate_candidates(args: argparse.Namespace) -> list[Candidate]:
    root_z_values = _parse_float_list(args.root_z_values)
    if args.use_current_default_pose:
        return [
            Candidate(
                index=-1,
                name="current_default_pose",
                root_z=root_z_values[0],
                joint_pos={},
            )
        ]

    crouch_arm_modes = _parse_str_list(args.crouch_arm_modes)
    measured_pose = _load_measured_pose_json(args.measured_pose_json)
    measured_modes = {"measured", "measured_torso", "measured_arms", "measured_right_arm"}
    valid_arm_modes = {"default", "zero", "ready"} | measured_modes
    for arm_mode in crouch_arm_modes:
        is_blend_mode = arm_mode.startswith("measured_right_arm_blend_")
        if arm_mode not in valid_arm_modes and not is_blend_mode:
            raise ValueError(f"Unsupported crouch arm mode: {arm_mode}")
        if (arm_mode in measured_modes or is_blend_mode) and measured_pose is None:
            raise ValueError("measured crouch arm modes require --measured_pose_json.")
        if is_blend_mode:
            _parse_blend_alpha(arm_mode, "measured_right_arm_blend")
    priority_presets = [
        ("zero_leg", 0.0, 0.0, 0.0),
        ("soft_straight", -0.05, 0.12, -0.07),
        ("light_crouch", -0.12, 0.25, -0.14),
    ]
    crouch_presets = [
        ("mild", -0.18, 0.38, -0.20),
        ("mid", -0.25, 0.50, -0.27),
        ("deep", -0.32, 0.62, -0.34),
        ("deep2", -0.38, 0.72, -0.40),
    ]
    waist_pitch_values = _parse_float_list(args.waist_pitch_values) if args.waist_pitch_values else [0.0]
    widths = _parse_float_list(args.stance_width_values) if args.stance_width_values else [0.00, 0.06, 0.10, 0.14]
    toe_out_values = _parse_float_list(args.toe_out_values) if args.toe_out_values else [0.0, 0.05, -0.05]
    if args.hip_pitch_values or args.knee_values or args.ankle_pitch_values:
        hip_pitch_values = _parse_float_list(args.hip_pitch_values) if args.hip_pitch_values else [-0.18, -0.25]
        knee_values = _parse_float_list(args.knee_values) if args.knee_values else [0.38, 0.50]
        ankle_pitch_values = (
            _parse_float_list(args.ankle_pitch_values) if args.ankle_pitch_values else [-0.20, -0.27]
        )
        crouch_presets = [
            (f"custom_hp{hip_pitch:+.2f}_kn{knee:+.2f}_ap{ankle_pitch:+.2f}", hip_pitch, knee, ankle_pitch)
            for hip_pitch in hip_pitch_values
            for knee in knee_values
            for ankle_pitch in ankle_pitch_values
        ]

    candidates: list[Candidate] = []

    baseline = _base_pose("ready", measured_pose)
    baseline["waist_pitch_joint"] = waist_pitch_values[0]
    baseline.update(
        _leg_pose(
            hip_pitch=-0.20,
            knee=0.42,
            ankle_pitch=-0.23,
            left_hip_roll=0.0,
            right_hip_roll=0.0,
            toe_out=0.0,
        )
    )
    candidates.append(Candidate(index=0, name="baseline_t1_like_ready_arms", root_z=0.90, joint_pos=baseline))

    idx = 1
    priority_arm_modes = ["default", "zero", "ready"]
    if measured_pose is not None:
        priority_arm_modes.extend(["measured", "measured_torso", "measured_arms", "measured_right_arm"])
    for root_z in root_z_values:
        for preset_name, hip_pitch, knee, ankle_pitch in priority_presets:
            for arm_mode in priority_arm_modes:
                pose = _base_pose(arm_mode, measured_pose)
                pose.update(
                    _leg_pose(
                        hip_pitch=hip_pitch,
                        knee=knee,
                        ankle_pitch=ankle_pitch,
                        left_hip_roll=0.0,
                        right_hip_roll=0.0,
                        toe_out=0.0,
                    )
                )
                name = f"{preset_name}_{arm_mode}_arms_z{root_z:.2f}"
                candidates.append(Candidate(index=idx, name=name, root_z=root_z, joint_pos=pose))
                idx += 1

    if measured_pose is not None:
        for root_z in root_z_values:
            candidates.append(
                Candidate(
                    index=idx,
                    name=f"measured_full_z{root_z:.2f}",
                    root_z=root_z,
                    joint_pos=dict(measured_pose),
                )
            )
            idx += 1

    priority_count = len(candidates)
    for root_z in root_z_values:
        for crouch_name, hip_pitch, knee, ankle_pitch in crouch_presets:
            for arm_mode in crouch_arm_modes:
                for waist_pitch in waist_pitch_values:
                    for width in widths:
                        roll_signs = (1.0,) if math.isclose(width, 0.0) else (1.0, -1.0)
                        for sign in roll_signs:
                            for toe_out in toe_out_values:
                                pose = _base_pose(arm_mode, measured_pose)
                                pose["waist_pitch_joint"] = waist_pitch
                                pose.update(
                                    _leg_pose(
                                        hip_pitch=hip_pitch,
                                        knee=knee,
                                        ankle_pitch=ankle_pitch,
                                        left_hip_roll=sign * width,
                                        right_hip_roll=-sign * width,
                                        toe_out=toe_out,
                                    )
                                )
                                name = (
                                    f"{crouch_name}_{arm_mode}_arms_z{root_z:.2f}"
                                    f"_waist{waist_pitch:+.2f}_w{sign * width:+.2f}_toe{toe_out:+.2f}"
                                )
                                candidates.append(Candidate(index=idx, name=name, root_z=root_z, joint_pos=pose))
                                idx += 1

    if args.candidate_index is not None:
        selected = [candidate for candidate in candidates if candidate.index == args.candidate_index]
        if not selected:
            raise ValueError(f"No generated candidate has index {args.candidate_index}.")
        return selected

    if args.max_candidates > 0 and len(candidates) > args.max_candidates:
        keep = candidates[: min(priority_count, args.max_candidates)]
        remaining = candidates[priority_count:]
        count = max(0, args.max_candidates - len(keep))
        if count > 0:
            sample_ids = torch.linspace(0, len(remaining) - 1, count).round().long().tolist()
            keep.extend(remaining[i] for i in sample_ids)
        candidates = keep

    return candidates


def _configure_env(env_cfg, args: argparse.Namespace, num_envs: int) -> None:
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.scene.max_episode_length_s = 999999999.0
    env_cfg.scene.seed = int(args.seed)

    env_cfg.noise.add_noise = True
    for attr in (
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
        env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
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
    env_cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.domain_rand.events.reset_locomotion_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.domain_rand.events.reset_manipulation_joints.params["velocity_range"] = (0.0, 0.0)

    env_cfg.ball.ball_speed_x_range = (0.0, 0.0)
    env_cfg.ball.ball_speed_y_range = (0.0, 0.0)
    env_cfg.ball.ball_speed_z_range = (0.0, 0.0)
    env_cfg.ball.ball_pos_y_range = (0.0, 0.0)
    env_cfg.ball.ball_max_eposide_length = 999999999.0
    env_cfg.ball.ball_reset_repeat = 1
    env_cfg.ball.max_serve_per_episode = 1_000_000

    _apply_pd_scales(env_cfg.scene.robot, args)


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


def _set_camera_view(env, args: argparse.Namespace) -> None:
    eye = torch.tensor(_parse_vec3(args.camera_eye), dtype=torch.float32, device=env.device)
    target = torch.tensor(_parse_vec3(args.camera_target), dtype=torch.float32, device=env.device)
    if env.scene.env_origins.numel() > 0:
        origin = env.scene.env_origins[0]
        eye = eye + origin
        target = target + origin
    env.sim.set_camera_view(eye.detach().cpu().tolist(), target.detach().cpu().tolist())


def _update_visualization(env, args: argparse.Namespace, simulation_app=None, *, force_camera: bool = False) -> None:
    if force_camera:
        _set_camera_view(env, args)
    env.sim.render()
    if simulation_app is not None and hasattr(simulation_app, "update"):
        simulation_app.update()
    if args.visualize_sleep > 0.0:
        time.sleep(args.visualize_sleep)


def _candidate_joint_tensor(env, candidates: list[Candidate]) -> torch.Tensor:
    joint_pos = env.robot.data.default_joint_pos[: len(candidates)].clone()
    name_to_id = {name: idx for idx, name in enumerate(env.robot.joint_names)}
    for env_id, candidate in enumerate(candidates):
        for joint_name, value in candidate.joint_pos.items():
            if joint_name in name_to_id:
                joint_pos[env_id, name_to_id[joint_name]] = float(value)
    limits = env.robot.data.soft_joint_pos_limits[: len(candidates)]
    return joint_pos.clamp(limits[..., 0], limits[..., 1])


def _apply_candidates(env, candidates: list[Candidate]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    env_ids = torch.arange(len(candidates), device=env.device)
    joint_pos = _candidate_joint_tensor(env, candidates)
    joint_vel = torch.zeros_like(joint_pos)

    root_pose = torch.cat([env.robot.data.root_pos_w, env.robot.data.root_quat_w], dim=-1)[: len(candidates)].clone()
    root_z = torch.tensor([candidate.root_z for candidate in candidates], device=env.device, dtype=root_pose.dtype)
    root_pose[:, 2] = env.scene.env_origins[: len(candidates), 2] + root_z
    root_velocity = torch.zeros((len(candidates), 6), device=env.device, dtype=root_pose.dtype)

    env.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.robot.set_joint_position_target(joint_pos[:, env.action_joint_ids], env.action_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    default_targets = env.robot.data.default_joint_pos[: len(candidates), env.action_joint_ids]
    target_actions = (joint_pos[:, env.action_joint_ids] - default_targets) / env.action_scale
    target_actions = torch.clamp(target_actions, -env.clip_actions, env.clip_actions)
    return target_actions, root_pose[:, :3].clone(), joint_pos


def _feet_contact(env, threshold: float) -> torch.Tensor:
    forces = env.contact_sensor.data.net_forces_w_history[:, :, env.feet_cfg.body_ids, :]
    return torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > threshold


def _score_results(results: list[Result], trial_steps: int) -> None:
    for result in results:
        survival_ratio = result.survival_steps / max(1, trial_steps)
        result.score = (
            1000.0 * survival_ratio
            + 90.0 * result.both_feet_contact_ratio
            - 180.0 * result.max_abs_roll_pitch
            - 180.0 * result.max_root_z_drift
            - 80.0 * result.max_foot_slip
            - 25.0 * result.min_paddle_future_dist
        )
        if result.reset_seen:
            result.score -= 250.0


def _write_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_index",
                "name",
                "score",
                "survival_steps",
                "survived",
                "reset_seen",
                "root_z",
                "final_root_z",
                "max_abs_roll_pitch",
                "max_root_z_drift",
                "max_foot_slip",
                "both_feet_contact_ratio",
                "min_paddle_future_dist",
                "final_roll",
                "final_pitch",
                "min_pitch",
                "max_pitch",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.candidate.index,
                    result.candidate.name,
                    f"{result.score:.6f}",
                    result.survival_steps,
                    int(result.survived),
                    int(result.reset_seen),
                    f"{result.candidate.root_z:.6f}",
                    f"{result.final_root_z:.6f}",
                    f"{result.max_abs_roll_pitch:.6f}",
                    f"{result.max_root_z_drift:.6f}",
                    f"{result.max_foot_slip:.6f}",
                    f"{result.both_feet_contact_ratio:.6f}",
                    f"{result.min_paddle_future_dist:.6f}",
                    f"{result.final_roll:.6f}",
                    f"{result.final_pitch:.6f}",
                    f"{result.min_pitch:.6f}",
                    f"{result.max_pitch:.6f}",
                ]
            )


def _print_top_results(results: list[Result], top_k: int, trial_steps: int) -> None:
    print("\n[A3 Standing Calibration] Top candidates")
    print(
        "rank  id    score    steps  ok  reset  rollpitch  z_drift  foot_slip  "
        "contact  final_pitch  paddle_future  name"
    )
    for rank, result in enumerate(results[:top_k], start=1):
        print(
            f"{rank:>4}  {result.candidate.index:>3}  {result.score:>8.2f}  "
            f"{result.survival_steps:>5}/{trial_steps:<5}  "
            f"{int(result.survived):>2}  {int(result.reset_seen):>5}  "
            f"{result.max_abs_roll_pitch:>9.4f}  {result.max_root_z_drift:>7.4f}  "
            f"{result.max_foot_slip:>9.4f}  {result.both_feet_contact_ratio:>7.3f}  "
            f"{result.final_pitch:>11.4f}  "
            f"{result.min_paddle_future_dist:>13.4f}  {result.candidate.name}"
        )

    best = results[0]
    print("\n[A3 Standing Calibration] Best candidate config block")
    print(f"# candidate_index={best.candidate.index}, name={best.candidate.name}")
    print(f"# root_z={best.candidate.root_z:.6f}")
    print("joint_pos={")
    for name in sorted(best.candidate.joint_pos):
        print(f'    "{name}": {best.candidate.joint_pos[name]:.6f},')
    print("}")


def _evaluate(env, candidates: list[Candidate], args: argparse.Namespace, simulation_app=None) -> list[Result]:
    actions, initial_root_pos, _ = _apply_candidates(env, candidates)
    num_envs = len(candidates)
    device = env.device

    foot_body_ids, _ = env.robot.find_bodies(env.cfg.robot.feet_body_names, preserve_order=True)
    initial_feet_xy = env.robot.data.body_pos_w[:num_envs, foot_body_ids, :2].detach().clone()

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    survival_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    reset_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
    max_abs_roll_pitch = torch.zeros(num_envs, dtype=torch.float, device=device)
    max_root_z_drift = torch.zeros(num_envs, dtype=torch.float, device=device)
    max_foot_slip = torch.zeros(num_envs, dtype=torch.float, device=device)
    both_feet_contact_steps = torch.zeros(num_envs, dtype=torch.float, device=device)
    min_paddle_future_dist = torch.full((num_envs,), float("inf"), dtype=torch.float, device=device)
    min_pitch = torch.full((num_envs,), float("inf"), dtype=torch.float, device=device)
    max_pitch = torch.full((num_envs,), -float("inf"), dtype=torch.float, device=device)

    for step in range(1, args.trial_steps + 1):
        with torch.inference_mode():
            _, _, reset_buf, _ = env.step(actions)

            active_before = alive.clone()
            roll, pitch, _ = _euler_xyz_from_quat_wxyz(env.robot.data.root_quat_w[:num_envs])
            abs_roll_pitch = torch.maximum(torch.abs(roll), torch.abs(pitch))
            root_z_drift = torch.abs(env.robot.data.root_pos_w[:num_envs, 2] - initial_root_pos[:, 2])
            feet_contact = _feet_contact(env, args.contact_threshold)[:num_envs]
            both_feet_contact = feet_contact.all(dim=1)
            feet_xy = env.robot.data.body_pos_w[:num_envs, foot_body_ids, :2]
            foot_slip = torch.linalg.norm(feet_xy - initial_feet_xy, dim=-1).max(dim=1)[0]
            paddle_env = env.paddle_touch_point[:num_envs] - env.scene.env_origins[:num_envs]
            paddle_future_dist = torch.linalg.norm(paddle_env - env.ball_future_pose[:num_envs], dim=1)

            max_abs_roll_pitch = torch.maximum(max_abs_roll_pitch, torch.where(active_before, abs_roll_pitch, 0.0))
            max_root_z_drift = torch.maximum(max_root_z_drift, torch.where(active_before, root_z_drift, 0.0))
            max_foot_slip = torch.maximum(max_foot_slip, torch.where(active_before, foot_slip, 0.0))
            min_pitch = torch.minimum(min_pitch, torch.where(active_before, pitch, min_pitch))
            max_pitch = torch.maximum(max_pitch, torch.where(active_before, pitch, max_pitch))
            min_paddle_future_dist = torch.minimum(
                min_paddle_future_dist,
                torch.where(active_before, paddle_future_dist, min_paddle_future_dist),
            )
            both_feet_contact_steps += (active_before & both_feet_contact).float()

            failed_now = (
                reset_buf[:num_envs]
                | (abs_roll_pitch > args.max_abs_roll_pitch)
                | (env.robot.data.root_pos_w[:num_envs, 2] < args.min_root_z)
                | (root_z_drift > args.max_root_z_drift)
            )
            reset_seen |= reset_buf[:num_envs]
            alive &= ~failed_now
            survival_steps += alive.long()

        if args.print_interval > 0 and (step == 1 or step % args.print_interval == 0):
            print(
                f"[A3 Standing Calibration] step {step}/{args.trial_steps}: "
                f"alive={int(alive.sum().detach().cpu())}/{num_envs}"
            )
        if args.keep_camera_interval > 0 and step % args.keep_camera_interval == 0:
            _update_visualization(env, args, simulation_app, force_camera=True)
        elif args.visualize_sleep > 0.0:
            _update_visualization(env, args, simulation_app)
        if args.stop_when_all_failed and not bool(alive.any()):
            break

    final_root_z = env.robot.data.root_pos_w[:num_envs, 2].detach().cpu()
    final_roll, final_pitch, _ = _euler_xyz_from_quat_wxyz(env.robot.data.root_quat_w[:num_envs])
    final_roll = final_roll.detach().cpu()
    final_pitch = final_pitch.detach().cpu()
    contact_ratio = both_feet_contact_steps / torch.clamp(survival_steps.float(), min=1.0)
    min_paddle_future_dist = torch.where(
        torch.isfinite(min_paddle_future_dist), min_paddle_future_dist, torch.zeros_like(min_paddle_future_dist)
    )
    min_pitch = torch.where(torch.isfinite(min_pitch), min_pitch, torch.zeros_like(min_pitch)).detach().cpu()
    max_pitch = torch.where(torch.isfinite(max_pitch), max_pitch, torch.zeros_like(max_pitch)).detach().cpu()

    results = []
    for i, candidate in enumerate(candidates):
        results.append(
            Result(
                candidate=candidate,
                score=0.0,
                survival_steps=int(survival_steps[i].detach().cpu()),
                survived=bool(alive[i].detach().cpu()) and int(survival_steps[i].detach().cpu()) >= args.trial_steps,
                reset_seen=bool(reset_seen[i].detach().cpu()),
                max_abs_roll_pitch=float(max_abs_roll_pitch[i].detach().cpu()),
                max_root_z_drift=float(max_root_z_drift[i].detach().cpu()),
                max_foot_slip=float(max_foot_slip[i].detach().cpu()),
                both_feet_contact_ratio=float(contact_ratio[i].detach().cpu()),
                min_paddle_future_dist=float(min_paddle_future_dist[i].detach().cpu()),
                final_root_z=float(final_root_z[i]),
                final_roll=float(final_roll[i]),
                final_pitch=float(final_pitch[i]),
                min_pitch=float(min_pitch[i]),
                max_pitch=float(max_pitch[i]),
            )
        )
    _score_results(results, args.trial_steps)
    results.sort(key=lambda item: item.score, reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-test A3 crouched table-tennis standing poses.")
    parser.add_argument("--task", type=str, default="a3_tt_eval", help="A3 task to launch.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=4.0)
    parser.add_argument("--trial_steps", type=int, default=250, help="Number of 50 Hz env steps per candidate.")
    parser.add_argument("--max_candidates", type=int, default=128, help="Maximum candidates to test in parallel.")
    parser.add_argument("--candidate_index", type=int, default=None, help="Test one printed candidate id.")
    parser.add_argument(
        "--use_current_default_pose",
        action="store_true",
        help="Visualize/evaluate the robot default joint pose from the current A3 asset config.",
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--print_interval", type=int, default=25)
    parser.add_argument("--stop_when_all_failed", action="store_true")
    parser.add_argument("--root_z_values", type=str, default="0.72,0.78,0.84,0.90,0.96,1.02")
    parser.add_argument("--waist_pitch_values", type=str, default="")
    parser.add_argument("--hip_pitch_values", type=str, default="")
    parser.add_argument("--knee_values", type=str, default="")
    parser.add_argument("--ankle_pitch_values", type=str, default="")
    parser.add_argument("--stance_width_values", type=str, default="")
    parser.add_argument("--toe_out_values", type=str, default="")
    parser.add_argument(
        "--crouch_arm_modes",
        type=str,
        default="ready",
        help=(
            "Comma-separated upper-body modes for crouch scan: default,zero,ready,measured,"
            "measured_torso,measured_arms,measured_right_arm,"
            "measured_right_arm_blend_<alpha>."
        ),
    )
    parser.add_argument(
        "--measured_pose_json",
        type=str,
        default="",
        help="Optional JSON with 'layout' and 'q'. Enables measured full-pose and measured upper-body candidates.",
    )
    parser.add_argument("--base_x", type=float, default=-0.26, help="Reset x offset relative to robot init pose.")
    parser.add_argument("--base_y", type=float, default=0.35, help="Reset y offset relative to robot init pose.")
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--max_abs_roll_pitch", type=float, default=0.55)
    parser.add_argument("--max_root_z_drift", type=float, default=0.22)
    parser.add_argument("--min_root_z", type=float, default=0.50)
    parser.add_argument("--contact_threshold", type=float, default=1.0)
    parser.add_argument("--waist_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--waist_damping_scale", type=float, default=1.0)
    parser.add_argument("--leg_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--leg_damping_scale", type=float, default=1.0)
    parser.add_argument("--feet_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--feet_damping_scale", type=float, default=1.0)
    parser.add_argument("--camera_eye", type=str, default="-3.2,-2.0,1.6")
    parser.add_argument("--camera_target", type=str, default="-1.8,0.35,0.85")
    parser.add_argument("--visualize_sleep", type=float, default=0.0)
    parser.add_argument("--keep_camera_interval", type=int, default=0)
    parser.add_argument("--warmup_render_steps", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="")

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    if not args_cli.task.startswith("a3_"):
        raise ValueError("This script is A3-only. Use --task=a3_tt or --task=a3_tt_eval.")

    candidates = _generate_candidates(args_cli)
    if not candidates:
        raise ValueError("No candidates generated.")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli, len(candidates))
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=bool(args_cli.headless))
    _set_camera_view(env, args_cli)
    for _ in range(args_cli.warmup_render_steps):
        _update_visualization(env, args_cli, simulation_app, force_camera=True)

    print("[A3 Standing Calibration] Started.")
    print(f"task: {args_cli.task}")
    print(f"candidates: {len(candidates)}")
    print(f"trial_steps: {args_cli.trial_steps}")

    try:
        results = _evaluate(env, candidates, args_cli, simulation_app)
        _print_top_results(results, min(args_cli.top_k, len(results)), args_cli.trial_steps)

        output_csv = args_cli.output_csv
        if not output_csv:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_csv = f"logs/a3_standing_calibration/{stamp}_results.csv"
        output_path = Path(output_csv)
        _write_csv(output_path, results)
        print(f"\n[A3 Standing Calibration] CSV written to: {output_path}")
    except KeyboardInterrupt:
        print("\n[A3 Standing Calibration] Interrupted.")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
