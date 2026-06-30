"""A3 table-tennis paddle pose calibration helper.

This script launches a single A3 table-tennis environment with a deterministic
serve and visual markers:

- green BallFuture marker: environment-computed ``ball_future_pose``
- yellow BallPred marker: current ``paddle_touch_point``

It does not load a policy and does not change training behavior.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

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


def _euler_xyz_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Isaac quaternions are stored as (w, x, y, z).
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


def _configure_calibration_env(env_cfg, args: argparse.Namespace, joint_overrides: dict[str, float]) -> None:
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.scene.env_spacing = float(args.env_spacing)
    env_cfg.scene.max_episode_length_s = 999999999.0
    env_cfg.scene.seed = int(args.seed)

    # Keep noise plumbing enabled because TTEnv.init_obs_buffer() expects it,
    # but set every scale to zero for deterministic calibration.
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

    # Keep the robot reset deterministic so pose changes are easy to compare.
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

    # Keep the serve deterministic by default.
    env_cfg.ball.ball_pos_y_range = (args.ball_y, args.ball_y)
    env_cfg.ball.ball_speed_x_range = (args.ball_vx, args.ball_vx)
    env_cfg.ball.ball_speed_y_range = (args.ball_vy, args.ball_vy)
    env_cfg.ball.ball_speed_z_range = (args.ball_vz, args.ball_vz)
    env_cfg.ball.ball_reset_repeat = 1
    env_cfg.ball.max_serve_per_episode = 1_000_000

    # Apply candidate right-arm default pose before the environment is built.
    env_cfg.scene.robot.init_state.joint_pos.update(joint_overrides)


def _joint_values(env, joint_names: tuple[str, ...]) -> dict[str, float]:
    values: dict[str, float] = {}
    for joint_name in joint_names:
        try:
            joint_id = env.robot.joint_names.index(joint_name)
        except ValueError:
            continue
        values[joint_name] = float(env.robot.data.joint_pos[0, joint_id].detach().cpu())
    return values


def _update_markers(env) -> None:
    # ``ball_future_pose`` is env-local, while the marker root pose is world-frame.
    env.ball_future_pose_vis = env.ball_future_pose + env.scene.env_origins
    env.update_ball_future_visual()

    # Reuse the yellow BallPred marker as the paddle-center marker for calibration.
    env.ball_prediction_vis = env.paddle_touch_point
    env.update_ball_pred_visual()


def _pin_base(env, root_pose: torch.Tensor, root_velocity: torch.Tensor) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    env.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _print_status(env, step: int, joint_names: tuple[str, ...], reset_count: int = 0) -> None:
    paddle_env = env.paddle_touch_point - env.scene.env_origins
    future_env = env.ball_future_pose
    dist_future = torch.linalg.norm(paddle_env - future_env, dim=1)
    ball_dist = torch.linalg.norm(env.ball.data.root_pos_w - env.paddle_touch_point, dim=1) - 0.02
    roll, pitch, yaw = _euler_xyz_from_quat_wxyz(env.robot.data.root_quat_w)
    paddle_quat = _normalize(env.robot.data.body_quat_w[:, env.paddle_body_id, :])
    local_x = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    local_y = torch.tensor([[0.0, 1.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    local_z = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)
    paddle_x_w = _normalize(_quat_apply_wxyz(paddle_quat, local_x))
    paddle_y_w = _normalize(_quat_apply_wxyz(paddle_quat, local_y))
    paddle_z_w = _normalize(_quat_apply_wxyz(paddle_quat, local_z))
    incoming_ball_dir = _normalize(-env.ball_linvel)
    plus_y_alignment = torch.sum(paddle_y_w * incoming_ball_dir, dim=-1)
    minus_y_alignment = -plus_y_alignment
    best_alignment = torch.maximum(plus_y_alignment, minus_y_alignment)

    print("\n[A3 Pose Calibration]")
    print(f"step: {step}")
    print(f"reset_count: {reset_count}")
    print(f"paddle_body: {env.paddle_body_name}")
    print(f"base_pos_w:             {_format_vec(env.robot.data.root_pos_w[0])}")
    print(
        "base_rpy(rad):          "
        f"[{float(roll[0].detach().cpu()): .4f}, "
        f"{float(pitch[0].detach().cpu()): .4f}, "
        f"{float(yaw[0].detach().cpu()): .4f}]"
    )
    print(f"ball_future_pose(env):   {_format_vec(future_env[0])}")
    print(f"paddle_touch_point(env): {_format_vec(paddle_env[0])}")
    print(f"ball_pos(env):           {_format_vec(env.ball_pos[0])}")
    print(f"ball_linvel(env):        {_format_vec(env.ball_linvel[0])}")
    print(f"incoming_ball_dir:       {_format_vec(incoming_ball_dir[0])}")
    print(f"paddle_local_x_w:        {_format_vec(paddle_x_w[0])}")
    print(f"paddle_face_+y_w:        {_format_vec(paddle_y_w[0])}")
    print(f"paddle_local_z_w:        {_format_vec(paddle_z_w[0])}")
    print(
        "face_alignment(+Y/-Y/best): "
        f"{float(plus_y_alignment[0].detach().cpu()): .4f} / "
        f"{float(minus_y_alignment[0].detach().cpu()): .4f} / "
        f"{float(best_alignment[0].detach().cpu()): .4f}"
    )
    print(f"paddle_to_future_dist:   {float(dist_future[0].detach().cpu()):.4f} m")
    print(f"paddle_to_ball_dist:     {float(ball_dist[0].detach().cpu()):.4f} m")
    print(f"ball_future_t:           {float(env.ball_future_t[0].detach().cpu()):.4f} s")
    print("right_arm_joint_pos:")
    for name, value in _joint_values(env, joint_names).items():
        print(f"  {name}: {value:.6f}")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize and measure A3 ping-pong paddle pre-pose.")
    parser.add_argument("--task", type=str, default="a3_tt_eval", help="A3 task to launch. Defaults to a3_tt_eval.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. Keep 1 for calibration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=5.0)
    parser.add_argument("--print_interval", type=int, default=25, help="Print every N env steps.")
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps. 0 means run until interrupted.")
    parser.add_argument("--no_visual_markers", action="store_true", help="Disable green/yellow marker updates.")
    parser.add_argument(
        "--pin_base",
        action="store_true",
        help="Keep robot root pose fixed during calibration. Debug only; not used for training.",
    )

    parser.add_argument("--base_x", type=float, default=-0.26, help="Reset x offset relative to robot init pose.")
    parser.add_argument("--base_y", type=float, default=0.35, help="Reset y offset relative to robot init pose.")
    parser.add_argument("--base_yaw", type=float, default=0.0, help="Reset yaw offset.")
    parser.add_argument("--ball_y", type=float, default=0.0)
    parser.add_argument("--ball_vx", type=float, default=-5.85)
    parser.add_argument("--ball_vy", type=float, default=-0.20)
    parser.add_argument("--ball_vz", type=float, default=1.70)
    parser.add_argument("--right_arm_noise", type=float, default=0.0, help="Deterministic offset added to reset right arm.")

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

    if not args_cli.task.startswith("a3_"):
        raise ValueError("This calibration script is A3-only. Use --task=a3_tt or --task=a3_tt_eval.")

    joint_overrides = _parse_joint_overrides(args_cli.joint)
    _add_optional_override(joint_overrides, "right_shoulder_pitch_joint", args_cli.rsp)
    _add_optional_override(joint_overrides, "right_shoulder_roll_joint", args_cli.rsr)
    _add_optional_override(joint_overrides, "right_shoulder_yaw_joint", args_cli.rsy)
    _add_optional_override(joint_overrides, "right_elbow_joint", args_cli.re)
    _add_optional_override(joint_overrides, "right_wrist_roll_joint", args_cli.rwr)
    _add_optional_override(joint_overrides, "right_wrist_pitch_joint", args_cli.rwp)
    _add_optional_override(joint_overrides, "right_wrist_yaw_joint", args_cli.rwy)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401

    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    _configure_calibration_env(env_cfg, args_cli, joint_overrides)
    env_class = task_registry.get_task_class(args_cli.task)

    # Livestream sets AppLauncher headless internally, but env markers should still update.
    env = env_class(env_cfg, headless=bool(args_cli.no_visual_markers))
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    pinned_root_pose = torch.cat([env.robot.data.root_pos_w, env.robot.data.root_quat_w], dim=-1).detach().clone()
    pinned_root_velocity = torch.zeros((env.num_envs, 6), device=env.device)

    print("[A3 Pose Calibration] Started.")
    print(f"task: {args_cli.task}")
    print("visual markers: green=ball_future_pose, yellow=paddle_touch_point")
    print("joint overrides:")
    if joint_overrides:
        for name, value in joint_overrides.items():
            print(f"  {name}: {value:.6f}")
    else:
        print("  <none, using configured A3 defaults>")

    step = 0
    reset_count = 0
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if args_cli.pin_base:
                    _pin_base(env, pinned_root_pose, pinned_root_velocity)
                _, _, reset_buf, _ = env.step(actions)
                reset_count += int(reset_buf.sum().detach().cpu())
                if args_cli.pin_base:
                    _pin_base(env, pinned_root_pose, pinned_root_velocity)
                step += 1

                if not args_cli.no_visual_markers:
                    _update_markers(env)

                if step == 1 or step % max(1, args_cli.print_interval) == 0:
                    _print_status(env, step, RIGHT_ARM_JOINTS, reset_count)

                if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                    break
    except KeyboardInterrupt:
        print("\n[A3 Pose Calibration] Interrupted.")
    finally:
        _print_status(env, step, RIGHT_ARM_JOINTS, reset_count)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
