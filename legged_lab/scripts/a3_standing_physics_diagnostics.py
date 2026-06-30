"""A3 standing physics diagnostics.

This A3-only helper diagnoses standing before table-tennis training:

1. pinned-root contact scan across root heights;
2. free-base rollout across root heights;
3. temporary PD multipliers for waist/legs/feet.

It does not modify training configuration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from isaaclab.app import AppLauncher


DEFAULT_ROOT_Z_VALUES = "0.72,0.78,0.84,0.90,0.96,1.02,1.08"


@dataclass
class ScanResult:
    phase: str
    root_z: float
    pose_name: str
    steps: int
    survival_steps: int
    reset_seen: bool
    mean_left_fz: float
    mean_right_fz: float
    max_left_force: float
    max_right_force: float
    both_feet_contact_ratio: float
    left_contact_ratio: float
    right_contact_ratio: float
    max_abs_roll_pitch: float
    max_root_z_drift: float
    max_foot_slip: float
    final_root_z: float
    final_left_foot_z: float
    final_right_foot_z: float
    first_env_reset_step: int
    min_robot_rel_x: float
    max_robot_rel_x: float
    min_robot_rel_y: float
    max_robot_rel_y: float
    min_robot_rel_z: float
    max_robot_rel_z: float


def _parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected a comma-separated list of floats.")
    return [float(item) for item in items]


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


def _leg_pose(hip_pitch: float, knee: float, ankle_pitch: float) -> dict[str, float]:
    return {
        "left_hip_pitch_joint": hip_pitch,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": knee,
        "left_ankle_pitch_joint": ankle_pitch,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": hip_pitch,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": knee,
        "right_ankle_pitch_joint": ankle_pitch,
        "right_ankle_roll_joint": 0.0,
    }


def _ready_arms() -> dict[str, float]:
    return {
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


def _zero_arms() -> dict[str, float]:
    return {
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


def _pose_overrides(name: str) -> dict[str, float]:
    if name == "current":
        return {}
    if name == "zero_leg":
        return _leg_pose(0.0, 0.0, 0.0)
    if name == "light_crouch":
        return _leg_pose(-0.12, 0.25, -0.14)
    if name == "t1_like":
        return _leg_pose(-0.20, 0.42, -0.23)
    if name == "ready_light_crouch":
        pose = _leg_pose(-0.12, 0.25, -0.14)
        pose.update(_ready_arms())
        return pose
    if name == "ready_t1_like":
        pose = _leg_pose(-0.20, 0.42, -0.23)
        pose.update(_ready_arms())
        return pose
    if name == "zero_light_crouch":
        pose = _leg_pose(-0.12, 0.25, -0.14)
        pose.update(_zero_arms())
        return pose
    if name == "zero_t1_like":
        pose = _leg_pose(-0.20, 0.42, -0.23)
        pose.update(_zero_arms())
        return pose
    raise ValueError(
        f"Unsupported --pose {name!r}. Valid: current, zero_leg, light_crouch, "
        "t1_like, ready_light_crouch, ready_t1_like, zero_light_crouch, zero_t1_like."
    )


def _scale_value(value: Any, scale: float) -> Any:
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
        env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (
            args.friction,
            args.friction,
        )
        env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (
            args.friction,
            args.friction,
        )
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


def _build_joint_state(env, pose_overrides: dict[str, float]) -> torch.Tensor:
    joint_pos = env.robot.data.default_joint_pos.clone()
    name_to_id = {name: idx for idx, name in enumerate(env.robot.joint_names)}
    for joint_name, value in pose_overrides.items():
        if joint_name in name_to_id:
            joint_pos[:, name_to_id[joint_name]] = float(value)
    limits = env.robot.data.soft_joint_pos_limits
    return joint_pos.clamp(limits[..., 0], limits[..., 1])


def _apply_pose(
    env,
    root_z_values: list[float],
    pose_overrides: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    env_ids = torch.arange(env.num_envs, device=env.device)
    joint_pos = _build_joint_state(env, pose_overrides)
    joint_vel = torch.zeros_like(joint_pos)

    root_pose = torch.cat([env.robot.data.root_pos_w, env.robot.data.root_quat_w], dim=-1).clone()
    root_z = torch.tensor(root_z_values, dtype=root_pose.dtype, device=env.device)
    root_pose[:, 2] = env.scene.env_origins[:, 2] + root_z
    root_velocity = torch.zeros((env.num_envs, 6), dtype=root_pose.dtype, device=env.device)

    env.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.robot.set_joint_position_target(joint_pos[:, env.action_joint_ids], env.action_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    default_targets = env.robot.data.default_joint_pos[:, env.action_joint_ids]
    actions = (joint_pos[:, env.action_joint_ids] - default_targets) / env.action_scale
    actions = torch.clamp(actions, -env.clip_actions, env.clip_actions)
    return actions, root_pose, joint_pos


def _pin_root(env, root_pose: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    zero_velocity = torch.zeros((env.num_envs, 6), dtype=root_pose.dtype, device=env.device)
    env.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def _contact_flags(env, threshold: float) -> torch.Tensor:
    forces = env.contact_sensor.data.net_forces_w_history[:, :, env.feet_cfg.body_ids, :]
    force_norm = torch.linalg.norm(forces, dim=-1).max(dim=1)[0]
    return force_norm > threshold


def _latest_foot_forces(env) -> torch.Tensor:
    return env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, :]


def _robot_pos_in_table_frame(env) -> torch.Tensor:
    if hasattr(env, "robot_pos"):
        return env.robot_pos
    return env.robot.data.root_link_pos_w - env.table.data.root_link_pos_w


def _collect_phase(
    env,
    root_z_values: list[float],
    pose_name: str,
    pose_overrides: dict[str, float],
    args: argparse.Namespace,
    *,
    phase: str,
    pinned: bool,
    steps: int,
) -> list[ScanResult]:
    actions, root_pose, _ = _apply_pose(env, root_z_values, pose_overrides)
    foot_body_ids, _ = env.robot.find_bodies(env.cfg.robot.feet_body_names, preserve_order=True)
    initial_feet_xy = env.robot.data.body_pos_w[:, foot_body_ids, :2].detach().clone()
    initial_root_z = root_pose[:, 2].detach().clone()

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    survival_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    max_abs_roll_pitch = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    max_root_z_drift = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    max_foot_slip = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    both_contact_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    left_contact_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    right_contact_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    sum_left_fz = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    sum_right_fz = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    max_left_force = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    max_right_force = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    first_env_reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    inf = torch.tensor(float("inf"), dtype=torch.float, device=env.device)
    min_robot_pos = torch.full((env.num_envs, 3), inf, dtype=torch.float, device=env.device)
    max_robot_pos = torch.full((env.num_envs, 3), -inf, dtype=torch.float, device=env.device)

    for step in range(1, steps + 1):
        with torch.inference_mode():
            if pinned:
                _pin_root(env, root_pose)
            _, _, reset_buf, _ = env.step(actions)
            if pinned:
                _pin_root(env, root_pose)

            active_before = alive.clone()
            roll, pitch, _ = _euler_xyz_from_quat_wxyz(env.robot.data.root_quat_w)
            abs_roll_pitch = torch.maximum(torch.abs(roll), torch.abs(pitch))
            root_z_drift = torch.abs(env.robot.data.root_pos_w[:, 2] - initial_root_z)
            foot_contact = _contact_flags(env, args.contact_threshold)
            both_contact = foot_contact.all(dim=1)
            foot_forces = _latest_foot_forces(env)
            left_force = torch.linalg.norm(foot_forces[:, 0, :], dim=-1)
            right_force = torch.linalg.norm(foot_forces[:, 1, :], dim=-1)
            left_fz = torch.clamp(foot_forces[:, 0, 2], min=0.0)
            right_fz = torch.clamp(foot_forces[:, 1, 2], min=0.0)
            feet_xy = env.robot.data.body_pos_w[:, foot_body_ids, :2]
            foot_slip = torch.linalg.norm(feet_xy - initial_feet_xy, dim=-1).max(dim=1)[0]
            robot_pos = _robot_pos_in_table_frame(env)

            max_abs_roll_pitch = torch.maximum(max_abs_roll_pitch, torch.where(active_before, abs_roll_pitch, 0.0))
            max_root_z_drift = torch.maximum(max_root_z_drift, torch.where(active_before, root_z_drift, 0.0))
            max_foot_slip = torch.maximum(max_foot_slip, torch.where(active_before, foot_slip, 0.0))
            max_left_force = torch.maximum(max_left_force, torch.where(active_before, left_force, 0.0))
            max_right_force = torch.maximum(max_right_force, torch.where(active_before, right_force, 0.0))
            sum_left_fz += torch.where(active_before, left_fz, 0.0)
            sum_right_fz += torch.where(active_before, right_fz, 0.0)
            left_contact_steps += (active_before & foot_contact[:, 0]).float()
            right_contact_steps += (active_before & foot_contact[:, 1]).float()
            both_contact_steps += (active_before & both_contact).float()
            active_mask = active_before.unsqueeze(-1)
            min_robot_pos = torch.minimum(min_robot_pos, torch.where(active_mask, robot_pos, inf))
            max_robot_pos = torch.maximum(max_robot_pos, torch.where(active_mask, robot_pos, -inf))
            new_env_reset = active_before & reset_buf & (first_env_reset_step < 0)
            first_env_reset_step = torch.where(
                new_env_reset,
                torch.full_like(first_env_reset_step, step),
                first_env_reset_step,
            )

            failed_now = (
                reset_buf
                | (abs_roll_pitch > args.max_abs_roll_pitch)
                | (env.robot.data.root_pos_w[:, 2] < args.min_root_z)
                | (root_z_drift > args.max_root_z_drift)
            )
            reset_seen |= reset_buf
            alive &= ~failed_now
            survival_steps += alive.long()

        if args.print_interval > 0 and (step == 1 or step % args.print_interval == 0):
            print(
                f"[A3 Standing Physics] {phase} step {step}/{steps}: "
                f"alive={int(alive.sum().detach().cpu())}/{env.num_envs}"
            )

    denom = torch.clamp(survival_steps.float(), min=1.0)
    final_root_z = env.robot.data.root_pos_w[:, 2].detach().cpu()
    final_foot_z = env.robot.data.body_pos_w[:, foot_body_ids, 2].detach().cpu()
    min_robot_pos_cpu = min_robot_pos.detach().cpu()
    max_robot_pos_cpu = max_robot_pos.detach().cpu()

    results: list[ScanResult] = []
    for i, root_z in enumerate(root_z_values):
        results.append(
            ScanResult(
                phase=phase,
                root_z=root_z,
                pose_name=pose_name,
                steps=steps,
                survival_steps=int(survival_steps[i].detach().cpu()),
                reset_seen=bool(reset_seen[i].detach().cpu()),
                mean_left_fz=float((sum_left_fz[i] / denom[i]).detach().cpu()),
                mean_right_fz=float((sum_right_fz[i] / denom[i]).detach().cpu()),
                max_left_force=float(max_left_force[i].detach().cpu()),
                max_right_force=float(max_right_force[i].detach().cpu()),
                both_feet_contact_ratio=float((both_contact_steps[i] / denom[i]).detach().cpu()),
                left_contact_ratio=float((left_contact_steps[i] / denom[i]).detach().cpu()),
                right_contact_ratio=float((right_contact_steps[i] / denom[i]).detach().cpu()),
                max_abs_roll_pitch=float(max_abs_roll_pitch[i].detach().cpu()),
                max_root_z_drift=float(max_root_z_drift[i].detach().cpu()),
                max_foot_slip=float(max_foot_slip[i].detach().cpu()),
                final_root_z=float(final_root_z[i]),
                final_left_foot_z=float(final_foot_z[i, 0]),
                final_right_foot_z=float(final_foot_z[i, 1]),
                first_env_reset_step=int(first_env_reset_step[i].detach().cpu()),
                min_robot_rel_x=float(min_robot_pos_cpu[i, 0]),
                max_robot_rel_x=float(max_robot_pos_cpu[i, 0]),
                min_robot_rel_y=float(min_robot_pos_cpu[i, 1]),
                max_robot_rel_y=float(max_robot_pos_cpu[i, 1]),
                min_robot_rel_z=float(min_robot_pos_cpu[i, 2]),
                max_robot_rel_z=float(max_robot_pos_cpu[i, 2]),
            )
        )
    return results


def _print_results(title: str, results: list[ScanResult]) -> None:
    print(f"\n[A3 Standing Physics] {title}")
    print(
        "phase        root_z  steps       reset  contact  Lfz     Rfz     "
        "rollpitch  z_drift  foot_slip  rst_step  rel_x_rng      rel_y_rng      rel_z_min"
    )
    for result in results:
        print(
            f"{result.phase:<11} {result.root_z:>6.3f}  "
            f"{result.survival_steps:>4}/{result.steps:<4}  "
            f"{int(result.reset_seen):>5}  {result.both_feet_contact_ratio:>7.3f}  "
            f"{result.mean_left_fz:>6.1f}  {result.mean_right_fz:>6.1f}  "
            f"{result.max_abs_roll_pitch:>9.4f}  {result.max_root_z_drift:>7.4f}  "
            f"{result.max_foot_slip:>9.4f}  {result.first_env_reset_step:>8}  "
            f"{result.min_robot_rel_x:>5.2f}..{result.max_robot_rel_x:<5.2f}  "
            f"{result.min_robot_rel_y:>5.2f}..{result.max_robot_rel_y:<5.2f}  "
            f"{result.min_robot_rel_z:>9.4f}"
        )


def _write_csv(path: Path, results: list[ScanResult], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "pose",
                "root_z",
                "steps",
                "survival_steps",
                "reset_seen",
                "mean_left_fz",
                "mean_right_fz",
                "max_left_force",
                "max_right_force",
                "both_feet_contact_ratio",
                "left_contact_ratio",
                "right_contact_ratio",
                "max_abs_roll_pitch",
                "max_root_z_drift",
                "max_foot_slip",
                "final_root_z",
                "final_left_foot_z",
                "final_right_foot_z",
                "first_env_reset_step",
                "min_robot_rel_x",
                "max_robot_rel_x",
                "min_robot_rel_y",
                "max_robot_rel_y",
                "min_robot_rel_z",
                "max_robot_rel_z",
                "waist_stiffness_scale",
                "waist_damping_scale",
                "leg_stiffness_scale",
                "leg_damping_scale",
                "feet_stiffness_scale",
                "feet_damping_scale",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.phase,
                    result.pose_name,
                    f"{result.root_z:.6f}",
                    result.steps,
                    result.survival_steps,
                    int(result.reset_seen),
                    f"{result.mean_left_fz:.6f}",
                    f"{result.mean_right_fz:.6f}",
                    f"{result.max_left_force:.6f}",
                    f"{result.max_right_force:.6f}",
                    f"{result.both_feet_contact_ratio:.6f}",
                    f"{result.left_contact_ratio:.6f}",
                    f"{result.right_contact_ratio:.6f}",
                    f"{result.max_abs_roll_pitch:.6f}",
                    f"{result.max_root_z_drift:.6f}",
                    f"{result.max_foot_slip:.6f}",
                    f"{result.final_root_z:.6f}",
                    f"{result.final_left_foot_z:.6f}",
                    f"{result.final_right_foot_z:.6f}",
                    result.first_env_reset_step,
                    f"{result.min_robot_rel_x:.6f}",
                    f"{result.max_robot_rel_x:.6f}",
                    f"{result.min_robot_rel_y:.6f}",
                    f"{result.max_robot_rel_y:.6f}",
                    f"{result.min_robot_rel_z:.6f}",
                    f"{result.max_robot_rel_z:.6f}",
                    f"{args.waist_stiffness_scale:.6f}",
                    f"{args.waist_damping_scale:.6f}",
                    f"{args.leg_stiffness_scale:.6f}",
                    f"{args.leg_damping_scale:.6f}",
                    f"{args.feet_stiffness_scale:.6f}",
                    f"{args.feet_damping_scale:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose A3 standing contact/root height/PD physics.")
    parser.add_argument("--task", type=str, default="a3_tt_eval")
    parser.add_argument("--pose", type=str, default="current")
    parser.add_argument("--root_z_values", type=str, default=DEFAULT_ROOT_Z_VALUES)
    parser.add_argument("--phase", type=str, default="pinned", choices=["pinned", "free"])
    parser.add_argument("--contact_steps", type=int, default=50)
    parser.add_argument("--free_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=4.0)
    parser.add_argument("--print_interval", type=int, default=25)
    parser.add_argument("--base_x", type=float, default=-0.26)
    parser.add_argument("--base_y", type=float, default=0.35)
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--contact_threshold", type=float, default=1.0)
    parser.add_argument("--max_abs_roll_pitch", type=float, default=0.55)
    parser.add_argument("--max_root_z_drift", type=float, default=0.22)
    parser.add_argument("--min_root_z", type=float, default=0.50)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--waist_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--waist_damping_scale", type=float, default=1.0)
    parser.add_argument("--leg_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--leg_damping_scale", type=float, default=1.0)
    parser.add_argument("--feet_stiffness_scale", type=float, default=1.0)
    parser.add_argument("--feet_damping_scale", type=float, default=1.0)
    parser.add_argument("--output_csv", type=str, default="")

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    if not args_cli.task.startswith("a3_"):
        raise ValueError("This script is A3-only. Use --task=a3_tt or --task=a3_tt_eval.")

    root_z_values = _parse_float_list(args_cli.root_z_values)
    pose_overrides = _pose_overrides(args_cli.pose)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli, len(root_z_values))
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=bool(args_cli.headless))

    print("[A3 Standing Physics] Started.")
    print(f"task: {args_cli.task}")
    print(f"pose: {args_cli.pose}")
    print(f"root_z_values: {root_z_values}")
    print(
        "pd scales: "
        f"waist=({args_cli.waist_stiffness_scale}, {args_cli.waist_damping_scale}), "
        f"legs=({args_cli.leg_stiffness_scale}, {args_cli.leg_damping_scale}), "
        f"feet=({args_cli.feet_stiffness_scale}, {args_cli.feet_damping_scale})"
    )

    try:
        if args_cli.phase == "pinned":
            results = _collect_phase(
                env,
                root_z_values,
                args_cli.pose,
                pose_overrides,
                args_cli,
                phase="pinned",
                pinned=True,
                steps=args_cli.contact_steps,
            )
            _print_results("Pinned-root contact scan", results)
        else:
            results = _collect_phase(
                env,
                root_z_values,
                args_cli.pose,
                pose_overrides,
                args_cli,
                phase="free",
                pinned=False,
                steps=args_cli.free_steps,
            )
            _print_results("Free-base rollout", results)

        output_csv = args_cli.output_csv
        if not output_csv:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_csv = f"logs/a3_standing_physics/{stamp}_{args_cli.pose}_{args_cli.phase}_results.csv"
        output_path = Path(output_csv)
        _write_csv(output_path, results, args_cli)
        print(f"\n[A3 Standing Physics] CSV written to: {output_path}")
    except KeyboardInterrupt:
        print("\n[A3 Standing Physics] Interrupted.")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
