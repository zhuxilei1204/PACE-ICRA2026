"""Diagnose a trained A3 Stage-1 policy with first-episode rollout stats."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


def _configure_clean_stage1(env_cfg, args: argparse.Namespace) -> None:
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.scene.seed = int(args.seed)
    if args.max_episode_length_s is not None:
        env_cfg.scene.max_episode_length_s = float(args.max_episode_length_s)

    if args.clean_eval:
        env_cfg.noise.add_noise = False
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


def _robot_pos(env) -> torch.Tensor:
    if hasattr(env, "robot") and hasattr(env, "table"):
        return env.robot.data.root_link_pos_w - env.table.data.root_link_pos_w
    if hasattr(env, "robot_pos"):
        return env.robot_pos
    return env.robot.data.root_link_pos_w - env.scene.env_origins


def _tilt(env) -> torch.Tensor:
    gravity_xy = env.robot.data.projected_gravity_b[:, :2]
    return torch.sqrt(torch.sum(torch.square(gravity_xy), dim=1) + 1e-12)


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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean().detach().cpu())


def _action_scale_tensor(env) -> torch.Tensor:
    scale = torch.as_tensor(env.action_scale, dtype=torch.float, device=env.device)
    if scale.ndim == 0:
        scale = scale.expand(env.num_actions)
    return scale.reshape(-1)


def _format_joint_rows(
    env,
    action_sum: torch.Tensor,
    action_abs_sum: torch.Tensor,
    action_square_sum: torch.Tensor,
    action_max_abs_joint: torch.Tensor,
    action_stat_count: float,
    top_joints: int,
) -> list[dict[str, float | int | str]]:
    denom = max(float(action_stat_count), 1.0)
    action_scale = _action_scale_tensor(env)
    mean = action_sum / denom
    abs_mean = action_abs_sum / denom
    rms = torch.sqrt(torch.clamp(action_square_sum / denom, min=0.0))
    target_offset_mean = mean * action_scale
    target_offset_abs_mean = abs_mean * torch.abs(action_scale)
    joint_names = list(getattr(env, "action_joint_names", [f"action_{i}" for i in range(env.num_actions)]))

    rows = []
    for i, name in enumerate(joint_names):
        rows.append(
            {
                "rank": 0,
                "action_index": i,
                "joint_name": name,
                "action_scale": float(action_scale[i].detach().cpu()),
                "action_mean": float(mean[i].detach().cpu()),
                "action_abs_mean": float(abs_mean[i].detach().cpu()),
                "action_rms": float(rms[i].detach().cpu()),
                "action_max_abs": float(action_max_abs_joint[i].detach().cpu()),
                "target_offset_mean": float(target_offset_mean[i].detach().cpu()),
                "target_offset_abs_mean": float(target_offset_abs_mean[i].detach().cpu()),
            }
        )

    rows.sort(key=lambda row: (abs(float(row["target_offset_abs_mean"])), abs(float(row["action_abs_mean"]))), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[: max(int(top_joints), 0)] if top_joints > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out an A3 Stage-1 policy and summarize first-episode drift.")
    parser.add_argument("--task", type=str, default="a3_tt_stage1_balance_move")
    parser.add_argument("--load_run", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=82)
    parser.add_argument("--env_spacing", type=float, default=4.0)
    parser.add_argument("--base_x", type=float, default=0.16)
    parser.add_argument("--base_y", type=float, default=0.35)
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--target_x", type=float, default=None)
    parser.add_argument("--target_y", type=float, default=None)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--continue_after_reset", action="store_true")
    parser.add_argument("--max_episode_length_s", type=float, default=None)
    parser.add_argument("--clean_eval", action="store_true", default=True)
    parser.add_argument("--no_clean_eval", action="store_false", dest="clean_eval")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--output_joint_csv", type=str, default="")
    parser.add_argument("--top_joints", type=int, default=12)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab_tasks.utils import get_checkpoint_path
    from rsl_rl.runners import OnPolicyRunner

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    _configure_clean_stage1(env_cfg, args)
    agent_cfg.load_run = args.load_run
    agent_cfg.load_checkpoint = args.checkpoint
    agent_cfg.seed = int(args.seed)
    env_cfg.scene.seed = int(args.seed)

    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, headless=bool(args.headless))

    try:
        log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, args.load_run, args.checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        policy_module = runner.alg.policy.to(env.device)

        obs, _ = env.get_observations()
        pos0 = _robot_pos(env).detach().clone()
        if args.target_x is None or args.target_y is None:
            if hasattr(env, "fixed_target_xy"):
                target_xy = env.fixed_target_xy.detach().clone()
            else:
                target_x, target_y = env.cfg.robot.future_invalid_robot_xy
                target_xy = torch.tensor(
                    [float(target_x), float(target_y)], dtype=torch.float, device=env.device
                ).expand(env.num_envs, 2)
        else:
            target_xy = torch.tensor(
                [float(args.target_x), float(args.target_y)], dtype=torch.float, device=env.device
            ).expand(env.num_envs, 2)

        inf = torch.tensor(float("inf"), device=env.device)
        min_pos = pos0.clone()
        max_pos = pos0.clone()
        end_pos = pos0.clone()
        max_tilt = _tilt(env).detach().clone()
        min_tilt = max_tilt.clone()
        roll0, pitch0, _ = _root_roll_pitch_yaw(env)
        gravity_xy0 = _projected_gravity_xy(env).detach().clone()
        min_roll = roll0.detach().clone()
        max_roll = roll0.detach().clone()
        end_roll = roll0.detach().clone()
        min_pitch = pitch0.detach().clone()
        max_pitch = pitch0.detach().clone()
        end_pitch = pitch0.detach().clone()
        max_abs_roll = torch.abs(roll0.detach().clone())
        max_abs_pitch = torch.abs(pitch0.detach().clone())
        end_gravity_xy = gravity_xy0.clone()
        max_abs_gravity_xy = torch.abs(gravity_xy0)
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        first_reset = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        alive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        progress_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        away_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        speed_count = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_l2_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_abs_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_max_abs = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_count = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_joint_sum = torch.zeros(env.num_actions, dtype=torch.float, device=env.device)
        action_joint_abs_sum = torch.zeros(env.num_actions, dtype=torch.float, device=env.device)
        action_joint_square_sum = torch.zeros(env.num_actions, dtype=torch.float, device=env.device)
        action_joint_max_abs = torch.zeros(env.num_actions, dtype=torch.float, device=env.device)
        action_joint_stat_count = 0.0
        low_z_reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        flat_reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_posture_reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        timeout_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        low_z_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        flat_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        bad_posture_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        timeout_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        for step in range(1, int(args.steps) + 1):
            with torch.inference_mode():
                stat_mask = torch.ones_like(active) if args.continue_after_reset else active
                before_pos = _robot_pos(env).detach().clone()
                before_tilt = _tilt(env).detach().clone()
                before_roll, before_pitch, _ = _root_roll_pitch_yaw(env)
                before_roll = before_roll.detach()
                before_pitch = before_pitch.detach()
                before_gravity_xy = _projected_gravity_xy(env).detach().clone()
                velocity_xy = env.robot.data.root_lin_vel_w[:, :2].detach()
                offset = target_xy - before_pos[:, :2]
                direction = offset / torch.clamp(torch.linalg.norm(offset, dim=1, keepdim=True), min=1.0e-6)
                progress_speed = torch.sum(velocity_xy * direction, dim=1)

                stat_float = stat_mask.float()
                progress_sum += torch.clamp(progress_speed, min=0.0) * stat_float
                away_sum += torch.clamp(-progress_speed, min=0.0) * stat_float
                speed_count += stat_float

                if args.stochastic:
                    actions = policy_module.act(obs)
                else:
                    actions = policy(obs)
                action_l2 = torch.sum(torch.square(actions), dim=1)
                action_l1_mean = torch.mean(torch.abs(actions), dim=1)
                action_l2_sum += action_l2 * stat_float
                action_abs_sum += action_l1_mean * stat_float
                action_max_abs = torch.maximum(action_max_abs, torch.max(torch.abs(actions), dim=1).values * stat_float)
                action_count += stat_float
                if bool(stat_mask.any()):
                    active_actions = actions[stat_mask]
                    action_joint_sum += active_actions.sum(dim=0)
                    action_joint_abs_sum += torch.abs(active_actions).sum(dim=0)
                    action_joint_square_sum += torch.square(active_actions).sum(dim=0)
                    action_joint_max_abs = torch.maximum(action_joint_max_abs, torch.max(torch.abs(active_actions), dim=0).values)
                    action_joint_stat_count += float(active_actions.shape[0])
                obs, _, reset_buf, _ = env.step(actions)

                min_pos = torch.minimum(min_pos, torch.where(stat_mask.unsqueeze(1), before_pos, inf))
                max_pos = torch.maximum(max_pos, torch.where(stat_mask.unsqueeze(1), before_pos, -inf))
                min_tilt = torch.minimum(min_tilt, torch.where(stat_mask, before_tilt, inf))
                max_tilt = torch.maximum(max_tilt, torch.where(stat_mask, before_tilt, -inf))
                end_pos = torch.where(stat_mask.unsqueeze(1), before_pos, end_pos)
                min_roll = torch.minimum(min_roll, torch.where(stat_mask, before_roll, inf))
                max_roll = torch.maximum(max_roll, torch.where(stat_mask, before_roll, -inf))
                min_pitch = torch.minimum(min_pitch, torch.where(stat_mask, before_pitch, inf))
                max_pitch = torch.maximum(max_pitch, torch.where(stat_mask, before_pitch, -inf))
                max_abs_roll = torch.maximum(max_abs_roll, torch.abs(before_roll) * stat_mask.float())
                max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(before_pitch) * stat_mask.float())
                max_abs_gravity_xy = torch.maximum(max_abs_gravity_xy, torch.abs(before_gravity_xy) * stat_mask.unsqueeze(1))
                end_roll = torch.where(stat_mask, before_roll, end_roll)
                end_pitch = torch.where(stat_mask, before_pitch, end_pitch)
                end_gravity_xy = torch.where(stat_mask.unsqueeze(1), before_gravity_xy, end_gravity_xy)

                low_z_now = stat_mask & _reset_cause(env, "reset_low_z_buf")
                flat_now = stat_mask & _reset_cause(env, "reset_flat_orientation_buf")
                bad_posture_now = stat_mask & _reset_cause(env, "reset_bad_posture_duration_buf")
                timeout_now = stat_mask & _reset_cause(env, "reset_episode_timeout_buf")
                new_reset = stat_mask & reset_buf
                low_z_reset_seen |= low_z_now
                flat_reset_seen |= flat_now
                bad_posture_reset_seen |= bad_posture_now
                timeout_seen |= timeout_now
                reset_count += new_reset.long()
                low_z_reset_count += low_z_now.long()
                flat_reset_count += flat_now.long()
                bad_posture_reset_count += bad_posture_now.long()
                timeout_count += timeout_now.long()
                first_reset = torch.where(new_reset & (first_reset < 0), torch.full_like(first_reset, step), first_reset)
                alive_steps += (stat_mask & ~reset_buf).long()
                if not args.continue_after_reset:
                    active &= ~reset_buf
                if not args.continue_after_reset and not bool(active.any()):
                    break

        survived = first_reset < 0
        progress_mean = progress_sum / torch.clamp(speed_count, min=1.0)
        away_mean = away_sum / torch.clamp(speed_count, min=1.0)
        delta = end_pos - pos0
        target_error0 = torch.linalg.norm(pos0[:, :2] - target_xy, dim=1)
        target_error_end = torch.linalg.norm(end_pos[:, :2] - target_xy, dim=1)
        target_error_delta = target_error_end - target_error0
        action_l2_mean = action_l2_sum / torch.clamp(action_count, min=1.0)
        action_abs_mean = action_abs_sum / torch.clamp(action_count, min=1.0)

        rows = []
        for i in range(env.num_envs):
            rows.append(
                {
                    "env_id": i,
                    "survived": bool(survived[i].detach().cpu()),
                    "first_reset_step": int(first_reset[i].detach().cpu()),
                    "alive_steps": int(alive_steps[i].detach().cpu()),
                    "start_x": float(pos0[i, 0].detach().cpu()),
                    "start_y": float(pos0[i, 1].detach().cpu()),
                    "start_z": float(pos0[i, 2].detach().cpu()),
                    "start_roll": float(roll0[i].detach().cpu()),
                    "start_pitch": float(pitch0[i].detach().cpu()),
                    "target_x": float(target_xy[i, 0].detach().cpu()),
                    "target_y": float(target_xy[i, 1].detach().cpu()),
                    "end_x": float(end_pos[i, 0].detach().cpu()),
                    "end_y": float(end_pos[i, 1].detach().cpu()),
                    "end_z": float(end_pos[i, 2].detach().cpu()),
                    "end_roll": float(end_roll[i].detach().cpu()),
                    "end_pitch": float(end_pitch[i].detach().cpu()),
                    "end_gravity_x": float(end_gravity_xy[i, 0].detach().cpu()),
                    "end_gravity_y": float(end_gravity_xy[i, 1].detach().cpu()),
                    "dx": float(delta[i, 0].detach().cpu()),
                    "dy": float(delta[i, 1].detach().cpu()),
                    "dz": float(delta[i, 2].detach().cpu()),
                    "min_x": float(min_pos[i, 0].detach().cpu()),
                    "max_x": float(max_pos[i, 0].detach().cpu()),
                    "min_y": float(min_pos[i, 1].detach().cpu()),
                    "max_y": float(max_pos[i, 1].detach().cpu()),
                    "min_z": float(min_pos[i, 2].detach().cpu()),
                    "max_z": float(max_pos[i, 2].detach().cpu()),
                    "min_tilt": float(min_tilt[i].detach().cpu()),
                    "max_tilt": float(max_tilt[i].detach().cpu()),
                    "min_roll": float(min_roll[i].detach().cpu()),
                    "max_roll": float(max_roll[i].detach().cpu()),
                    "min_pitch": float(min_pitch[i].detach().cpu()),
                    "max_pitch": float(max_pitch[i].detach().cpu()),
                    "max_abs_roll": float(max_abs_roll[i].detach().cpu()),
                    "max_abs_pitch": float(max_abs_pitch[i].detach().cpu()),
                    "max_abs_gravity_x": float(max_abs_gravity_xy[i, 0].detach().cpu()),
                    "max_abs_gravity_y": float(max_abs_gravity_xy[i, 1].detach().cpu()),
                    "mean_progress_speed": float(progress_mean[i].detach().cpu()),
                    "mean_away_speed": float(away_mean[i].detach().cpu()),
                    "target_error_start": float(target_error0[i].detach().cpu()),
                    "target_error_end": float(target_error_end[i].detach().cpu()),
                    "target_error_delta": float(target_error_delta[i].detach().cpu()),
                    "action_l2_mean": float(action_l2_mean[i].detach().cpu()),
                    "action_abs_mean": float(action_abs_mean[i].detach().cpu()),
                    "action_max_abs": float(action_max_abs[i].detach().cpu()),
                    "low_z_reset_seen": bool(low_z_reset_seen[i].detach().cpu()),
                    "flat_reset_seen": bool(flat_reset_seen[i].detach().cpu()),
                    "bad_posture_reset_seen": bool(bad_posture_reset_seen[i].detach().cpu()),
                    "timeout_seen": bool(timeout_seen[i].detach().cpu()),
                    "reset_count": int(reset_count[i].detach().cpu()),
                    "low_z_reset_count": int(low_z_reset_count[i].detach().cpu()),
                    "flat_reset_count": int(flat_reset_count[i].detach().cpu()),
                    "bad_posture_reset_count": int(bad_posture_reset_count[i].detach().cpu()),
                    "timeout_count": int(timeout_count[i].detach().cpu()),
                }
            )

        active_all = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        print(f"[A3 Stage1 Policy Diagnostics] checkpoint={resume_path}")
        print(
            f"envs={env.num_envs} steps={args.steps} clean_eval={args.clean_eval} "
            f"stochastic={args.stochastic} continue_after_reset={args.continue_after_reset}"
        )
        print(
            "target_xy_range="
            f"x[{float(target_xy[:, 0].min().item()):+.3f}, {float(target_xy[:, 0].max().item()):+.3f}] "
            f"y[{float(target_xy[:, 1].min().item()):+.3f}, {float(target_xy[:, 1].max().item()):+.3f}]"
        )
        print(f"survived={int(survived.sum().item())}/{env.num_envs}")
        print(f"alive_steps_mean={float(alive_steps.float().mean().item()):.2f}")
        print(f"alive_steps_min={int(alive_steps.min().item())} max={int(alive_steps.max().item())}")
        print(f"dx_mean={_masked_mean(delta[:, 0], active_all):+.4f} dy_mean={_masked_mean(delta[:, 1], active_all):+.4f}")
        print(f"z_mean_end={_masked_mean(end_pos[:, 2], active_all):+.4f}")
        print(f"target_error_delta_mean={_masked_mean(target_error_delta, active_all):+.4f}")
        print(f"progress_speed_mean={_masked_mean(progress_mean, active_all):+.4f}")
        print(f"away_speed_mean={_masked_mean(away_mean, active_all):+.4f}")
        print(f"max_tilt_mean={_masked_mean(max_tilt, active_all):+.4f}")
        print(
            f"end_roll_mean={_masked_mean(end_roll, active_all):+.4f} "
            f"end_pitch_mean={_masked_mean(end_pitch, active_all):+.4f}"
        )
        print(
            f"max_abs_roll_mean={_masked_mean(max_abs_roll, active_all):+.4f} "
            f"max_abs_pitch_mean={_masked_mean(max_abs_pitch, active_all):+.4f}"
        )
        print(
            f"end_gravity_x_mean={_masked_mean(end_gravity_xy[:, 0], active_all):+.4f} "
            f"end_gravity_y_mean={_masked_mean(end_gravity_xy[:, 1], active_all):+.4f}"
        )
        print(
            f"max_abs_gravity_x_mean={_masked_mean(max_abs_gravity_xy[:, 0], active_all):+.4f} "
            f"max_abs_gravity_y_mean={_masked_mean(max_abs_gravity_xy[:, 1], active_all):+.4f}"
        )
        print(f"x_range_mean={_masked_mean(max_pos[:, 0] - min_pos[:, 0], active_all):+.4f}")
        print(f"y_range_mean={_masked_mean(max_pos[:, 1] - min_pos[:, 1], active_all):+.4f}")
        print(f"action_l2_mean={_masked_mean(action_l2_mean, active_all):+.6f}")
        print(f"action_abs_mean={_masked_mean(action_abs_mean, active_all):+.6f}")
        print(f"action_max_abs={float(action_max_abs.max().detach().cpu()):+.6f}")
        joint_rows = _format_joint_rows(
            env,
            action_joint_sum,
            action_joint_abs_sum,
            action_joint_square_sum,
            action_joint_max_abs,
            action_joint_stat_count,
            args.top_joints,
        )
        if joint_rows:
            print("top_action_joints_by_effective_offset:")
            for row in joint_rows:
                print(
                    f"  {int(row['rank']):02d} {row['joint_name']:<30} "
                    f"scale={row['action_scale']:+.3f} "
                    f"mean={row['action_mean']:+.5f} "
                    f"abs={row['action_abs_mean']:.5f} "
                    f"max={row['action_max_abs']:.5f} "
                    f"dq_mean={row['target_offset_mean']:+.5f} "
                    f"dq_abs={row['target_offset_abs_mean']:.5f}"
                )
        print(
            "reset_causes="
            f"low_z:{int(low_z_reset_seen.sum().item())} "
            f"flat:{int(flat_reset_seen.sum().item())} "
            f"bad_posture:{int(bad_posture_reset_seen.sum().item())} "
            f"timeout:{int(timeout_seen.sum().item())}"
        )
        print(
            "reset_counts="
            f"total:{int(reset_count.sum().item())} "
            f"low_z:{int(low_z_reset_count.sum().item())} "
            f"flat:{int(flat_reset_count.sum().item())} "
            f"bad_posture:{int(bad_posture_reset_count.sum().item())} "
            f"timeout:{int(timeout_count.sum().item())}"
        )

        if args.output_csv:
            path = Path(args.output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"csv={path}")
        if args.output_joint_csv:
            joint_path = Path(args.output_joint_csv)
            joint_path.parent.mkdir(parents=True, exist_ok=True)
            with joint_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
                writer.writeheader()
                writer.writerows(joint_rows)
            print(f"joint_csv={joint_path}")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
