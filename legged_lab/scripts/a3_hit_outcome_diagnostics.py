"""Diagnose A3 table-tennis hit outcomes for candidate right-arm poses.

This helper is A3-only and read-only with respect to training configuration.
It launches zero-action rollouts, optionally applies temporary right-arm default
joint overrides, and reports whether the first paddle hit sends the ball toward
the opponent table with enough net clearance.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry


RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _parse_joint_overrides(items: Iterable[str] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not items:
        return overrides
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --joint value {item!r}; expected name=value.")
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in RIGHT_ARM_JOINTS:
            valid = ", ".join(RIGHT_ARM_JOINTS)
            raise ValueError(f"Unsupported joint {name!r}. Valid A3 right-arm joints: {valid}")
        overrides[name] = float(value)
    return overrides


def _add_optional_override(overrides: dict[str, float], name: str, value: float | None) -> None:
    if value is not None:
        overrides[name] = float(value)


def _configure_env(env_cfg, args: argparse.Namespace, joint_overrides: dict[str, float]) -> None:
    policy_mode = bool(args.load_run or args.checkpoint)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = float(args.env_spacing)
    if not policy_mode or args.ignore_task_resets:
        env_cfg.scene.max_episode_length_s = 999999999.0
    env_cfg.scene.seed = int(args.seed)

    # Keep noise plumbing enabled because TTEnv.init_obs_buffer() expects it,
    # but zero every scale for deterministic diagnostics.
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
    env_cfg.domain_rand.perception_delay.enable = False
    env_cfg.domain_rand.action_delay.enable = False
    if env_cfg.domain_rand.events.add_base_mass is not None:
        env_cfg.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)

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
    env_cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (
        args.right_arm_noise,
        args.right_arm_noise,
    )
    env_cfg.domain_rand.events.reset_manipulation_joints.params["velocity_range"] = (0.0, 0.0)

    if not args.random_ball:
        env_cfg.ball.ball_pos_y_range = (args.ball_y, args.ball_y)
        env_cfg.ball.ball_speed_x_range = (args.ball_vx, args.ball_vx)
        env_cfg.ball.ball_speed_y_range = (args.ball_vy, args.ball_vy)
        env_cfg.ball.ball_speed_z_range = (args.ball_vz, args.ball_vz)
    if not policy_mode or args.ignore_task_resets:
        env_cfg.ball.ball_reset_repeat = 1
        env_cfg.ball.max_serve_per_episode = 1_000_000

    env_cfg.scene.robot.init_state.joint_pos.update(joint_overrides)


def _estimate_net_z(pos: torch.Tensor, vel: torch.Tensor, net_x: float = 0.0) -> torch.Tensor:
    x = pos[:, 0]
    z = pos[:, 2]
    vx = vel[:, 0]
    vz = vel[:, 2]
    valid = vx > 1e-5
    t_net = (float(net_x) - x) / torch.clamp(vx, min=1e-5)
    valid &= t_net >= 0.0
    z_at_net = z + vz * t_net - 0.5 * 9.81 * t_net * t_net
    return torch.where(valid, z_at_net, torch.full_like(z_at_net, float("nan")))


def _format_stat(values: torch.Tensor, mask: torch.Tensor) -> str:
    selected = values[mask & torch.isfinite(values)]
    if selected.numel() == 0:
        return "n/a"
    return (
        f"mean={float(selected.mean().detach().cpu()):.4f}, "
        f"min={float(selected.min().detach().cpu()):.4f}, "
        f"max={float(selected.max().detach().cpu()):.4f}"
    )


def _ratio(mask: torch.Tensor, total: int) -> str:
    count = int(mask.sum().detach().cpu())
    return f"{count}/{total} ({count / max(total, 1):.3f})"


def _write_csv(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_rows(env, metrics: dict[str, torch.Tensor]) -> list[dict[str, float | int | bool]]:
    rows: list[dict[str, float | int | bool]] = []
    for env_id in range(env.num_envs):
        rows.append(
            {
                "env_id": env_id,
                "hit": bool(metrics["first_hit_seen"][env_id].detach().cpu()),
                "hit_step": int(metrics["first_hit_step"][env_id].detach().cpu()),
                "hit_x": float(metrics["hit_pos"][env_id, 0].detach().cpu()),
                "hit_y": float(metrics["hit_pos"][env_id, 1].detach().cpu()),
                "hit_z": float(metrics["hit_pos"][env_id, 2].detach().cpu()),
                "hit_vx": float(metrics["hit_vel"][env_id, 0].detach().cpu()),
                "hit_vy": float(metrics["hit_vel"][env_id, 1].detach().cpu()),
                "hit_vz": float(metrics["hit_vel"][env_id, 2].detach().cpu()),
                "estimated_z_at_net": float(metrics["estimated_z_at_net"][env_id].detach().cpu()),
                "actual_crossed_net": bool(metrics["actual_crossed_net"][env_id].detach().cpu()),
                "actual_z_at_net": float(metrics["actual_z_at_net"][env_id].detach().cpu()),
                "opponent_table_after_hit": bool(metrics["opponent_table_after_hit"][env_id].detach().cpu()),
                "own_table_before_hit": bool(metrics["own_table_before_hit"][env_id].detach().cpu()),
                "own_table_after_hit": bool(metrics["own_table_after_hit"][env_id].detach().cpu()),
                "reset_seen": bool(metrics["reset_seen"][env_id].detach().cpu()),
                "reset_step": int(metrics["reset_step"][env_id].detach().cpu()),
                "reset_low_z": bool(metrics["reset_low_z"][env_id].detach().cpu()),
                "reset_x_low": bool(metrics["reset_x_low"][env_id].detach().cpu()),
                "reset_x_high": bool(metrics["reset_x_high"][env_id].detach().cpu()),
                "reset_y_low": bool(metrics["reset_y_low"][env_id].detach().cpu()),
                "reset_y_high": bool(metrics["reset_y_high"][env_id].detach().cpu()),
                "reset_timeout": bool(metrics["reset_timeout"][env_id].detach().cpu()),
                "reset_robot_x": float(metrics["reset_robot_pos"][env_id, 0].detach().cpu()),
                "reset_robot_y": float(metrics["reset_robot_pos"][env_id, 1].detach().cpu()),
                "reset_robot_z": float(metrics["reset_robot_pos"][env_id, 2].detach().cpu()),
            }
        )
    return rows


def _print_summary(
    env,
    args: argparse.Namespace,
    joint_overrides: dict[str, float],
    metrics: dict[str, torch.Tensor],
    resume_path: str | None,
) -> None:
    total = env.num_envs
    hit = metrics["first_hit_seen"]
    hit_forward = hit & (metrics["hit_vel"][:, 0] > 0.0)
    hit_up = hit & (metrics["hit_vel"][:, 2] > 0.0)
    estimated_clear = hit & torch.isfinite(metrics["estimated_z_at_net"]) & (metrics["estimated_z_at_net"] > args.net_z_target)
    actual_clear = (
        hit
        & metrics["actual_crossed_net"]
        & torch.isfinite(metrics["actual_z_at_net"])
        & (metrics["actual_z_at_net"] > args.net_z_target)
    )

    print("\n[A3 Hit Outcome Diagnostics]")
    print(f"task: {args.task}")
    print(f"num_envs: {total}")
    print(f"steps: {args.max_steps}")
    print(f"action_mode: {'policy' if resume_path else 'zero'}")
    if resume_path:
        print(f"checkpoint: {resume_path}")
    print(f"random_ball: {bool(args.random_ball)}")
    print(f"ignore_task_resets: {bool(args.ignore_task_resets)}")
    print(f"net_z_target: {args.net_z_target:.3f}")
    print("joint_overrides:")
    if joint_overrides:
        for name, value in joint_overrides.items():
            print(f"  {name}: {value:.6f}")
    else:
        print("  <none, using configured A3 defaults>")
    print("")
    print(f"first_hit:                 {_ratio(hit, total)}")
    print(f"hit_vx_positive:           {_ratio(hit_forward, total)}")
    print(f"hit_vz_positive:           {_ratio(hit_up, total)}")
    print(f"estimated_net_clear:       {_ratio(estimated_clear, total)}")
    print(f"actual_crossed_net:        {_ratio(metrics['actual_crossed_net'], total)}")
    print(f"actual_net_clear:          {_ratio(actual_clear, total)}")
    print(f"opponent_table_after_hit:  {_ratio(metrics['opponent_table_after_hit'], total)}")
    print(f"own_table_before_hit:      {_ratio(metrics['own_table_before_hit'], total)}")
    print(f"own_table_after_hit:       {_ratio(metrics['own_table_after_hit'], total)}")
    print(f"reset_seen:                {_ratio(metrics['reset_seen'], total)}")
    print(f"reset_low_z:               {_ratio(metrics['reset_low_z'], total)}")
    print(f"reset_x_low:               {_ratio(metrics['reset_x_low'], total)}")
    print(f"reset_x_high:              {_ratio(metrics['reset_x_high'], total)}")
    print(f"reset_y_low:               {_ratio(metrics['reset_y_low'], total)}")
    print(f"reset_y_high:              {_ratio(metrics['reset_y_high'], total)}")
    print(f"reset_timeout:             {_ratio(metrics['reset_timeout'], total)}")
    print("")
    print(f"first_hit_step:            {_format_stat(metrics['first_hit_step'].float(), hit)}")
    print(f"reset_step:                {_format_stat(metrics['reset_step'].float(), metrics['reset_seen'])}")
    print(f"reset_robot_x:             {_format_stat(metrics['reset_robot_pos'][:, 0], metrics['reset_seen'])}")
    print(f"reset_robot_y:             {_format_stat(metrics['reset_robot_pos'][:, 1], metrics['reset_seen'])}")
    print(f"reset_robot_z:             {_format_stat(metrics['reset_robot_pos'][:, 2], metrics['reset_seen'])}")
    print(f"hit_x:                     {_format_stat(metrics['hit_pos'][:, 0], hit)}")
    print(f"hit_y:                     {_format_stat(metrics['hit_pos'][:, 1], hit)}")
    print(f"hit_z:                     {_format_stat(metrics['hit_pos'][:, 2], hit)}")
    print(f"hit_vx:                    {_format_stat(metrics['hit_vel'][:, 0], hit)}")
    print(f"hit_vy:                    {_format_stat(metrics['hit_vel'][:, 1], hit)}")
    print(f"hit_vz:                    {_format_stat(metrics['hit_vel'][:, 2], hit)}")
    print(f"estimated_z_at_net:        {_format_stat(metrics['estimated_z_at_net'], hit)}")
    print(f"actual_z_at_net:           {_format_stat(metrics['actual_z_at_net'], metrics['actual_crossed_net'])}")
    print("", flush=True)


def _load_policy(env, agent_cfg, args: argparse.Namespace):
    if not args.load_run and not args.checkpoint:
        return None, None, None
    if not args.load_run or not args.checkpoint:
        raise ValueError("--load_run and --checkpoint must be provided together for policy diagnostics.")

    from isaaclab_tasks.utils import get_checkpoint_path

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, args.load_run, args.checkpoint)
    log_dir = os.path.dirname(resume_path)
    if args.predictor:
        from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner

        runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        from rsl_rl.rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    try:
        runner.eval_mode()
    except Exception:
        pass
    policy = runner.get_inference_policy(device=env.device)
    return runner, policy, resume_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose first-hit outcomes for A3 table tennis.")
    parser.add_argument("--task", type=str, default="a3_tt")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=2.5)
    parser.add_argument("--max_steps", type=int, default=140)
    parser.add_argument("--print_interval", type=int, default=0, help="Print rollout progress every N env steps.")
    parser.add_argument("--net_z_target", type=float, default=1.11)
    parser.add_argument(
        "--ignore_task_resets",
        action="store_true",
        help="Ignore task episode/serve reset limits. Default for zero-action mode; optional for policy mode.",
    )

    parser.add_argument("--base_x", type=float, default=-0.26)
    parser.add_argument("--base_y", type=float, default=0.35)
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--random_ball", action="store_true", help="Use the task's configured ball randomization ranges.")
    parser.add_argument("--ball_y", type=float, default=0.0)
    parser.add_argument("--ball_vx", type=float, default=-5.0)
    parser.add_argument("--ball_vy", type=float, default=-0.04)
    parser.add_argument("--ball_vz", type=float, default=1.5)
    parser.add_argument("--right_arm_noise", type=float, default=0.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--load_run", type=str, default=None, help="A3 run directory name under logs/a3_table_tennis.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file, e.g. model_119.pt.")
    parser.add_argument("--predictor", action="store_true", help="Load predictor-augmented runner and update ball_prediction.")

    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override an A3 right-arm default joint position. Can be repeated.",
    )
    parser.add_argument("--rsp", type=float, default=None, help="Alias for right_shoulder_pitch_joint.")
    parser.add_argument("--rsr", type=float, default=None, help="Alias for right_shoulder_roll_joint.")
    parser.add_argument("--rsy", type=float, default=None, help="Alias for right_shoulder_yaw_joint.")
    parser.add_argument("--re", type=float, default=None, help="Alias for right_elbow_joint.")
    parser.add_argument("--rwr", type=float, default=None, help="Alias for right_wrist_roll_joint.")
    parser.add_argument("--rwp", type=float, default=None, help="Alias for right_wrist_pitch_joint.")
    parser.add_argument("--rwy", type=float, default=None, help="Alias for right_wrist_yaw_joint.")

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    valid_tasks = {
        "a3_tt",
        "a3_tt_eval",
        "a3_tt_stable",
        "a3_tt_stable_eval",
        "a3_tt_stage4b",
        "a3_tt_stage4b_eval",
        "a3_tt_stage4c",
        "a3_tt_stage4c_eval",
        "a3_tt_stage4d",
        "a3_tt_stage4d_eval",
        "a3_tt_stage5_ready",
        "a3_tt_stage5_ready_eval",
        "a3_tt_stage5b",
        "a3_tt_stage5b_eval",
        "a3_tt_stage5c",
        "a3_tt_stage5c_eval",
        "a3_tt_stage4e",
        "a3_tt_stage4e_eval",
        "a3_tt_stage4f",
        "a3_tt_stage4f_eval",
        "a3_tt_stage4g",
        "a3_tt_stage4g_eval",
        "a3_tt_stage4h",
        "a3_tt_stage4h_eval",
    }
    if args_cli.task not in valid_tasks:
        raise ValueError(
            "This diagnostic script is A3-only. Use --task=a3_tt, --task=a3_tt_eval, "
            "--task=a3_tt_stable, --task=a3_tt_stable_eval, --task=a3_tt_stage4b, "
            "--task=a3_tt_stage4b_eval, --task=a3_tt_stage4c, --task=a3_tt_stage4c_eval, "
            "--task=a3_tt_stage4d, --task=a3_tt_stage4d_eval, --task=a3_tt_stage4e, "
            "--task=a3_tt_stage4e_eval, --task=a3_tt_stage5_ready, "
            "--task=a3_tt_stage5_ready_eval, --task=a3_tt_stage5b, --task=a3_tt_stage5b_eval, "
            "--task=a3_tt_stage5c, --task=a3_tt_stage5c_eval, "
            "--task=a3_tt_stage4f, --task=a3_tt_stage4f_eval, "
            "--task=a3_tt_stage4g, --task=a3_tt_stage4g_eval, --task=a3_tt_stage4h, "
            "or --task=a3_tt_stage4h_eval."
        )

    joint_overrides = _parse_joint_overrides(args_cli.joint)
    _add_optional_override(joint_overrides, "right_shoulder_pitch_joint", args_cli.rsp)
    _add_optional_override(joint_overrides, "right_shoulder_roll_joint", args_cli.rsr)
    _add_optional_override(joint_overrides, "right_shoulder_yaw_joint", args_cli.rsy)
    _add_optional_override(joint_overrides, "right_elbow_joint", args_cli.re)
    _add_optional_override(joint_overrides, "right_wrist_roll_joint", args_cli.rwr)
    _add_optional_override(joint_overrides, "right_wrist_pitch_joint", args_cli.rwp)
    _add_optional_override(joint_overrides, "right_wrist_yaw_joint", args_cli.rwy)

    print("[A3 Hit Outcome Diagnostics] Launching Isaac app...", flush=True)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    print("[A3 Hit Outcome Diagnostics] Loading task registry...", flush=True)
    import legged_lab.envs  # noqa: F401

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli, joint_overrides)
    env_class = task_registry.get_task_class(args_cli.task)
    print("[A3 Hit Outcome Diagnostics] Building environment...", flush=True)
    env = env_class(env_cfg, headless=True)
    runner, policy, resume_path = _load_policy(env, agent_cfg, args_cli)
    print("[A3 Hit Outcome Diagnostics] Running rollout...", flush=True)
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs = None
    if policy is not None:
        obs, _ = env.get_observations()

    first_hit_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_hit_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    hit_pos = torch.full((env.num_envs, 3), float("nan"), dtype=torch.float, device=env.device)
    hit_vel = torch.full((env.num_envs, 3), float("nan"), dtype=torch.float, device=env.device)
    estimated_z_at_net = torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device)
    actual_crossed_net = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    actual_z_at_net = torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device)
    opponent_table_after_hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    own_table_before_hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    own_table_after_hit_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    own_table_after_hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    reset_low_z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_x_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_x_high = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_y_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_y_high = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_timeout = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_robot_pos = torch.full((env.num_envs, 3), float("nan"), dtype=torch.float, device=env.device)

    prev_ball_pos = env.ball.data.root_pos_w.detach().clone() - env.scene.env_origins

    try:
        step = 0
        while simulation_app.is_running() and step < args_cli.max_steps:
            with torch.inference_mode():
                if policy is not None:
                    actions = policy(obs)
                obs, _, reset_buf, _ = env.step(actions)
                if args_cli.predictor and runner is not None:
                    try:
                        runner._record_ball_positions()
                        runner._maybe_predict_and_update_env()
                    except Exception:
                        pass
                step += 1
                if args_cli.print_interval > 0 and (step == 1 or step % args_cli.print_interval == 0):
                    print(
                        "[A3 Hit Outcome Diagnostics] "
                        f"step {step}/{args_cli.max_steps}, "
                        f"hits={int(first_hit_seen.sum().detach().cpu())}/{env.num_envs}, "
                        f"resets={int(reset_seen.sum().detach().cpu())}/{env.num_envs}",
                        flush=True,
                    )

                ball_pos = env.ball.data.root_pos_w.detach().clone() - env.scene.env_origins
                ball_vel = env.ball.data.root_lin_vel_w.detach().clone()
                own_table_now = getattr(
                    env,
                    "has_touch_own_table_just_now",
                    torch.zeros_like(first_hit_seen),
                )
                own_table_before_hit |= ~first_hit_seen & own_table_now
                new_hit = env.has_touch_paddle & ~first_hit_seen
                if new_hit.any():
                    first_hit_seen[new_hit] = True
                    first_hit_step[new_hit] = step
                    hit_pos[new_hit] = ball_pos[new_hit]
                    hit_vel[new_hit] = ball_vel[new_hit]
                    estimated_z_at_net[new_hit] = _estimate_net_z(ball_pos[new_hit], ball_vel[new_hit])

                active_after_hit = first_hit_seen
                if hasattr(env, "has_touch_opponent_table_just_now"):
                    opponent_table_after_hit |= active_after_hit & env.has_touch_opponent_table_just_now
                own_table_after_hit_armed |= active_after_hit & ~own_table_now
                own_table_after_hit |= own_table_after_hit_armed & own_table_now

                crossing = (
                    active_after_hit
                    & ~actual_crossed_net
                    & (prev_ball_pos[:, 0] < 0.0)
                    & (ball_pos[:, 0] >= 0.0)
                )
                if crossing.any():
                    denom = torch.clamp(ball_pos[crossing, 0] - prev_ball_pos[crossing, 0], min=1e-6)
                    alpha = torch.clamp((0.0 - prev_ball_pos[crossing, 0]) / denom, min=0.0, max=1.0)
                    actual_z_at_net[crossing] = prev_ball_pos[crossing, 2] + alpha * (
                        ball_pos[crossing, 2] - prev_ball_pos[crossing, 2]
                    )
                    actual_crossed_net[crossing] = True

                new_reset = reset_buf & ~reset_seen
                if new_reset.any():
                    robot_pos = env.robot_pos.detach().clone()
                    low_z = robot_pos[:, 2] < 0.50
                    x_low = robot_pos[:, 0] < -3.6
                    x_high = robot_pos[:, 0] > -1.35
                    y_low = robot_pos[:, 1] < -1.1
                    y_high = robot_pos[:, 1] > 1.1
                    timeout = env.time_out_buf.detach().clone() if hasattr(env, "time_out_buf") else torch.zeros_like(new_reset)

                    reset_seen[new_reset] = True
                    reset_step[new_reset] = step
                    reset_low_z[new_reset] = low_z[new_reset]
                    reset_x_low[new_reset] = x_low[new_reset]
                    reset_x_high[new_reset] = x_high[new_reset]
                    reset_y_low[new_reset] = y_low[new_reset]
                    reset_y_high[new_reset] = y_high[new_reset]
                    reset_timeout[new_reset] = timeout[new_reset]
                    reset_robot_pos[new_reset] = robot_pos[new_reset]

                prev_ball_pos = ball_pos
    finally:
        metrics = {
            "first_hit_seen": first_hit_seen,
            "first_hit_step": first_hit_step,
            "hit_pos": hit_pos,
            "hit_vel": hit_vel,
            "estimated_z_at_net": estimated_z_at_net,
            "actual_crossed_net": actual_crossed_net,
            "actual_z_at_net": actual_z_at_net,
            "opponent_table_after_hit": opponent_table_after_hit,
            "own_table_before_hit": own_table_before_hit,
            "own_table_after_hit": own_table_after_hit,
            "reset_seen": reset_seen,
            "reset_step": reset_step,
            "reset_low_z": reset_low_z,
            "reset_x_low": reset_x_low,
            "reset_x_high": reset_x_high,
            "reset_y_low": reset_y_low,
            "reset_y_high": reset_y_high,
            "reset_timeout": reset_timeout,
            "reset_robot_pos": reset_robot_pos,
        }
        _print_summary(env, args_cli, joint_overrides, metrics, resume_path)
        if args_cli.csv is not None:
            _write_csv(args_cli.csv, _build_rows(env, metrics))
            print(f"\nCSV written to: {args_cli.csv}", flush=True)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
