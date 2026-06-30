"""WebRTC-friendly TT environment probe.

This script is intentionally separate from train/eval/preview.  It keeps the
camera fixed, renders at a controlled pace, and updates the Kit app every step
so remote WebRTC clients do not go black when the simulation loop starts.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from isaaclab.app import AppLauncher


def _parse_vec3(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != 3:
        raise ValueError("Expected three comma-separated floats.")
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


def _set_camera_view(env, args: argparse.Namespace) -> None:
    eye = torch.tensor(_parse_vec3(args.camera_eye), dtype=torch.float32, device=env.device)
    target = torch.tensor(_parse_vec3(args.camera_target), dtype=torch.float32, device=env.device)
    if env.scene.env_origins.numel() > 0:
        origin = env.scene.env_origins[0]
        eye = eye + origin
        target = target + origin
    env.sim.set_camera_view(eye.detach().cpu().tolist(), target.detach().cpu().tolist())


def _render(env, simulation_app, args: argparse.Namespace, *, force_camera: bool = False) -> None:
    if force_camera:
        _set_camera_view(env, args)
    env.sim.render()
    if hasattr(simulation_app, "update"):
        simulation_app.update()
    if args.visualize_sleep > 0.0:
        time.sleep(args.visualize_sleep)


def _freeze_ball_if_requested(env_cfg, args: argparse.Namespace) -> None:
    if not args.freeze_ball or not hasattr(env_cfg, "ball"):
        return
    ball_cfg = env_cfg.ball
    for attr in ("ball_speed_x_range", "ball_speed_y_range", "ball_speed_z_range"):
        if hasattr(ball_cfg, attr):
            setattr(ball_cfg, attr, (0.0, 0.0))
    if hasattr(ball_cfg, "ball_pos_y_range"):
        ball_cfg.ball_pos_y_range = (0.0, 0.0)
    if hasattr(ball_cfg, "ball_max_eposide_length"):
        ball_cfg.ball_max_eposide_length = 999999999.0
    if hasattr(ball_cfg, "ball_reset_repeat"):
        ball_cfg.ball_reset_repeat = 1
    if hasattr(ball_cfg, "max_serve_per_episode"):
        ball_cfg.max_serve_per_episode = 1_000_000


def _make_actions(env, args: argparse.Namespace, step: int, current_actions: torch.Tensor) -> torch.Tensor:
    if args.mode == "zero":
        return torch.zeros_like(current_actions)
    if args.mode == "random":
        if step % max(1, args.action_hold_steps) == 1:
            return torch.clamp(args.action_std * torch.randn_like(current_actions), -1.0, 1.0)
        return current_actions
    if args.mode == "sine":
        phase = float(step) * args.sine_frequency
        action = args.action_std * torch.sin(
            phase + torch.arange(env.num_actions, device=env.device, dtype=torch.float32)
        )
        return action.unsqueeze(0).repeat(env.num_envs, 1)
    raise ValueError(f"Unsupported mode: {args.mode}")


def _load_policy(env, agent_cfg, args: argparse.Namespace):
    if args.mode != "policy":
        return None, None, None
    if not args.load_run or not args.checkpoint:
        raise ValueError("--mode=policy requires --load_run and --checkpoint.")

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
    parser = argparse.ArgumentParser(description="WebRTC-friendly TT task visual probe.")
    parser.add_argument("--task", type=str, default="t1_tt_eval")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial_steps", type=int, default=100000)
    parser.add_argument("--mode", type=str, default="zero", choices=["zero", "random", "sine", "policy"])
    parser.add_argument("--action_std", type=float, default=0.20)
    parser.add_argument("--action_hold_steps", type=int, default=5)
    parser.add_argument("--sine_frequency", type=float, default=0.05)
    parser.add_argument("--load_run", type=str, default=None, help="Run directory name under the task log root.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename, e.g. model_2549.pt.")
    parser.add_argument("--predictor", action="store_true", help="Load a predictor-augmented runner.")
    parser.add_argument(
        "--disable_predictor_update",
        action="store_true",
        help="Do not update predictor state each step in policy mode.",
    )
    parser.add_argument("--freeze_ball", action="store_true")
    parser.add_argument("--print_interval", type=int, default=50)
    parser.add_argument("--camera_eye", type=str, default="-3.2,-2.0,1.6")
    parser.add_argument("--camera_target", type=str, default="-1.8,0.35,0.85")
    parser.add_argument("--warmup_render_steps", type=int, default=30)
    parser.add_argument("--keep_camera_interval", type=int, default=1)
    parser.add_argument("--visualize_sleep", type=float, default=0.03)

    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = args_cli.seed
    _freeze_ball_if_requested(env_cfg, args_cli)

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, headless=False)
    _set_camera_view(env, args_cli)

    runner, policy, resume_path = _load_policy(env, agent_cfg, args_cli)

    for _ in range(args_cli.warmup_render_steps):
        _render(env, simulation_app, args_cli, force_camera=True)

    obs, _ = env.get_observations()
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    reset_count = 0

    print("[TT WebRTC Probe] Started.")
    print(f"task: {args_cli.task}")
    print(f"num_envs: {env.num_envs}")
    print(f"mode: {args_cli.mode}")
    print(f"freeze_ball: {args_cli.freeze_ball}")
    if resume_path is not None:
        print(f"checkpoint: {resume_path}")

    try:
        for step in range(1, args_cli.trial_steps + 1):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                if args_cli.mode == "policy":
                    actions = policy(obs)
                else:
                    actions = _make_actions(env, args_cli, step, actions)
                obs, _, reset_buf, _ = env.step(actions)
                if args_cli.mode == "policy" and args_cli.predictor and not args_cli.disable_predictor_update:
                    try:
                        runner._record_ball_positions()
                        runner._maybe_predict_and_update_env()
                    except Exception:
                        pass
                reset_count += int(reset_buf.sum().detach().cpu())

            if args_cli.keep_camera_interval > 0 and step % args_cli.keep_camera_interval == 0:
                _render(env, simulation_app, args_cli, force_camera=True)
            else:
                _render(env, simulation_app, args_cli)

            if args_cli.print_interval > 0 and (step == 1 or step % args_cli.print_interval == 0):
                roll, pitch, _ = _euler_xyz_from_quat_wxyz(env.robot.data.root_quat_w[:1])
                root_pos = env.robot.data.root_pos_w[0].detach().cpu().tolist()
                print(
                    "[TT WebRTC Probe] "
                    f"step={step}/{args_cli.trial_steps} "
                    f"resets={reset_count} "
                    f"root=({root_pos[0]:.3f},{root_pos[1]:.3f},{root_pos[2]:.3f}) "
                    f"roll={float(roll[0]):.3f} pitch={float(pitch[0]):.3f}"
                )
    except KeyboardInterrupt:
        print("\n[TT WebRTC Probe] Interrupted.")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
