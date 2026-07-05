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


def _robot_pos(env) -> torch.Tensor:
    if hasattr(env, "robot_pos"):
        return env.robot_pos
    return env.robot.data.root_link_pos_w - env.table.data.root_link_pos_w


def _tilt(env) -> torch.Tensor:
    gravity_xy = env.robot.data.projected_gravity_b[:, :2]
    return torch.sqrt(torch.sum(torch.square(gravity_xy), dim=1) + 1e-12)


def _reset_cause(env, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return value


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].float().mean().detach().cpu())


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
        low_z_reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        flat_reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        timeout_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        low_z_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        flat_reset_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        timeout_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        for step in range(1, int(args.steps) + 1):
            with torch.inference_mode():
                stat_mask = torch.ones_like(active) if args.continue_after_reset else active
                before_pos = _robot_pos(env).detach().clone()
                before_tilt = _tilt(env).detach().clone()
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
                obs, _, reset_buf, _ = env.step(actions)

                min_pos = torch.minimum(min_pos, torch.where(stat_mask.unsqueeze(1), before_pos, inf))
                max_pos = torch.maximum(max_pos, torch.where(stat_mask.unsqueeze(1), before_pos, -inf))
                min_tilt = torch.minimum(min_tilt, torch.where(stat_mask, before_tilt, inf))
                max_tilt = torch.maximum(max_tilt, torch.where(stat_mask, before_tilt, -inf))
                end_pos = torch.where(stat_mask.unsqueeze(1), before_pos, end_pos)

                low_z_now = stat_mask & _reset_cause(env, "reset_low_z_buf")
                flat_now = stat_mask & _reset_cause(env, "reset_flat_orientation_buf")
                timeout_now = stat_mask & _reset_cause(env, "reset_episode_timeout_buf")
                new_reset = stat_mask & reset_buf
                low_z_reset_seen |= low_z_now
                flat_reset_seen |= flat_now
                timeout_seen |= timeout_now
                reset_count += new_reset.long()
                low_z_reset_count += low_z_now.long()
                flat_reset_count += flat_now.long()
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
                    "target_x": float(target_xy[i, 0].detach().cpu()),
                    "target_y": float(target_xy[i, 1].detach().cpu()),
                    "end_x": float(end_pos[i, 0].detach().cpu()),
                    "end_y": float(end_pos[i, 1].detach().cpu()),
                    "end_z": float(end_pos[i, 2].detach().cpu()),
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
                    "timeout_seen": bool(timeout_seen[i].detach().cpu()),
                    "reset_count": int(reset_count[i].detach().cpu()),
                    "low_z_reset_count": int(low_z_reset_count[i].detach().cpu()),
                    "flat_reset_count": int(flat_reset_count[i].detach().cpu()),
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
        print(f"x_range_mean={_masked_mean(max_pos[:, 0] - min_pos[:, 0], active_all):+.4f}")
        print(f"y_range_mean={_masked_mean(max_pos[:, 1] - min_pos[:, 1], active_all):+.4f}")
        print(f"action_l2_mean={_masked_mean(action_l2_mean, active_all):+.6f}")
        print(f"action_abs_mean={_masked_mean(action_abs_mean, active_all):+.6f}")
        print(f"action_max_abs={float(action_max_abs.max().detach().cpu()):+.6f}")
        print(
            "reset_causes="
            f"low_z:{int(low_z_reset_seen.sum().item())} "
            f"flat:{int(flat_reset_seen.sum().item())} "
            f"timeout:{int(timeout_seen.sum().item())}"
        )
        print(
            "reset_counts="
            f"total:{int(reset_count.sum().item())} "
            f"low_z:{int(low_z_reset_count.sum().item())} "
            f"flat:{int(flat_reset_count.sum().item())} "
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
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
