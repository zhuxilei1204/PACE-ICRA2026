"""Diagnose TT paddle pose against the incoming ball path.

This script is read-only with respect to task configuration.  It launches one TT
environment, prints the paddle body axes, and reports which local axis best
faces the incoming ball direction.
"""

from __future__ import annotations

import argparse
import sys

import torch
from isaaclab.app import AppLauncher


def _format_vec(tensor: torch.Tensor) -> str:
    vals = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{float(v): .4f}" for v in vals) + "]"


def _normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=eps)


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_w = quat[:, :1]
    q_xyz = quat[:, 1:]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def _configure_env(env_cfg, args: argparse.Namespace) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 5.0
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

    env_cfg.ball.ball_pos_y_range = (args.ball_y, args.ball_y)
    env_cfg.ball.ball_speed_x_range = (args.ball_vx, args.ball_vx)
    env_cfg.ball.ball_speed_y_range = (args.ball_vy, args.ball_vy)
    env_cfg.ball.ball_speed_z_range = (args.ball_vz, args.ball_vz)
    env_cfg.ball.ball_reset_repeat = 1
    env_cfg.ball.max_serve_per_episode = 1_000_000


def _print_status(env, step: int, reset_count: int) -> None:
    paddle_pos = env.robot.data.body_pos_w[:, env.paddle_body_id, :]
    paddle_quat = _normalize(env.robot.data.body_quat_w[:, env.paddle_body_id, :])
    local_x = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    local_y = torch.tensor([[0.0, 1.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    local_z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    axes = {
        "+X": _normalize(_quat_apply_wxyz(paddle_quat, local_x)),
        "-X": -_normalize(_quat_apply_wxyz(paddle_quat, local_x)),
        "+Y": _normalize(_quat_apply_wxyz(paddle_quat, local_y)),
        "-Y": -_normalize(_quat_apply_wxyz(paddle_quat, local_y)),
        "+Z": _normalize(_quat_apply_wxyz(paddle_quat, local_z)),
        "-Z": -_normalize(_quat_apply_wxyz(paddle_quat, local_z)),
    }
    incoming_ball_dir = _normalize(-env.ball_linvel)
    alignments = {name: torch.sum(axis * incoming_ball_dir, dim=-1) for name, axis in axes.items()}
    best_axis_name = max(alignments, key=lambda name: float(alignments[name][0].detach().cpu()))

    paddle_env = env.paddle_touch_point - env.scene.env_origins
    dist_future = torch.linalg.norm(paddle_env - env.ball_future_pose, dim=1)
    offset = env.paddle_local_offset.unsqueeze(0) if env.paddle_local_offset.ndim == 1 else env.paddle_local_offset
    offset_w = _quat_apply_wxyz(paddle_quat, offset.expand_as(paddle_pos))

    print("\n[TT Paddle Pose Diagnostics]")
    print(f"task_paddle_body:        {env.paddle_body_name}")
    print(f"step:                    {step}")
    print(f"reset_count:             {reset_count}")
    print(f"root_pos_w:              {_format_vec(env.robot.data.root_pos_w[0])}")
    print(f"future_paddle_x_offset: {float(env.cfg.robot.future_paddle_x_offset): .4f}")
    print(f"future_paddle_y_offset: {float(env.cfg.robot.future_paddle_y_offset): .4f}")
    print(f"ball_future_pose(env):   {_format_vec(env.ball_future_pose[0])}")
    print(f"robot_future_pos(env):   {_format_vec(env.robot_future_pos[0])}")
    print(f"ball_pos(env):           {_format_vec(env.ball_pos[0])}")
    print(f"ball_linvel(env):        {_format_vec(env.ball_linvel[0])}")
    print(f"incoming_ball_dir:       {_format_vec(incoming_ball_dir[0])}")
    print(f"paddle_body_pos(env):    {_format_vec((paddle_pos - env.scene.env_origins)[0])}")
    print(f"paddle_local_offset:     {_format_vec(env.paddle_local_offset)}")
    print(f"paddle_offset_w:         {_format_vec(offset_w[0])}")
    print(f"paddle_touch_point(env): {_format_vec(paddle_env[0])}")
    for name in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
        print(f"axis_{name}_w:              {_format_vec(axes[name][0])}")
        print(f"align_{name}:              {float(alignments[name][0].detach().cpu()): .4f}")
    print(f"best_axis:               {best_axis_name}")
    print(f"best_alignment:          {float(alignments[best_axis_name][0].detach().cpu()):.4f}")
    print(f"paddle_to_future_dist:   {float(dist_future[0].detach().cpu()):.4f} m")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose TT paddle pose relative to ball path.")
    parser.add_argument("--task", type=str, default="t1_tt_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_x", type=float, default=0.0)
    parser.add_argument("--base_y", type=float, default=0.0)
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--ball_y", type=float, default=0.0)
    parser.add_argument("--ball_vx", type=float, default=-5.85)
    parser.add_argument("--ball_vy", type=float, default=-0.20)
    parser.add_argument("--ball_vz", type=float, default=1.70)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--print_interval", type=int, default=1)

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli)
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=True)
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    step = 0
    reset_count = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                _, _, reset_buf, _ = env.step(actions)
                reset_count += int(reset_buf.sum().detach().cpu())
                step += 1
            if step == 1 or step % max(1, args_cli.print_interval) == 0:
                _print_status(env, step, reset_count)
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
