# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import argparse
import os
import time
import numpy as np

import torch
from isaaclab.app import AppLauncher
# Use the Isaac Lab adapter for RSL-RL to match the env API
from rsl_rl.rsl_rl.runners import OnPolicyRunner
# from isaaclab_rl.rsl_rl import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train/Play an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--predictor", action="store_true", help="Use predictor-augmented runner for loading predictor weights and inference")
# recording / debugging flags
parser.add_argument(
    "--record_action",
    action="store_true",
    help="Record observations and actions during play; also plots obs[48:69].",
)
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=[-3.2, -2.0, 1.6],
    metavar=("X", "Y", "Z"),
    help="Viewport camera position for GUI/WebRTC play.",
)
parser.add_argument(
    "--camera_target",
    type=float,
    nargs=3,
    default=[-1.8, 0.35, 0.85],
    metavar=("X", "Y", "Z"),
    help="Viewport camera look-at target for GUI/WebRTC play.",
)
parser.add_argument(
    "--keep_camera_interval",
    type=int,
    default=1,
    help="Re-apply the viewport camera every N play steps in GUI/WebRTC mode. Use 0 to disable.",
)
parser.add_argument(
    "--visualize_sleep",
    type=float,
    default=None,
    help="Sleep duration after each rendered play step. Defaults to 0.03s for livestream, 0 for local GUI.",
)
parser.add_argument(
    "--skip_export",
    action="store_true",
    help="Skip predictor/JIT/ONNX export before play. Useful for WebRTC visualization.",
)
parser.add_argument(
    "--no_load_runner",
    action="store_true",
    help="Do not load checkpoint/runner/policy; useful for isolating WebRTC rendering.",
)
parser.add_argument(
    "--max_play_steps",
    type=int,
    default=0,
    help="Stop play after N environment steps. 0 means run until interrupted.",
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="policy",
    choices=["policy", "zero", "random", "sine"],
    help="Action source for play visualization.",
)
parser.add_argument("--action_std", type=float, default=0.20, help="Action amplitude for random/sine action modes.")
parser.add_argument("--action_hold_steps", type=int, default=5, help="Hold random actions for this many steps.")
parser.add_argument("--sine_frequency", type=float, default=0.05, help="Frequency for sine action mode.")
parser.add_argument(
    "--disable_predictor_update",
    action="store_true",
    help="Do not update predictor visualization state each play step.",
)
parser.add_argument(
    "--warmup_render_steps",
    type=int,
    default=None,
    help="Render this many frames before policy stepping. Defaults to 30 for livestream and 0 otherwise.",
)
# parser.add_argument("--check", type=str, default='model_.*.pt', help="checkpoint, defaul the lastest")
# parser.add_argument("--run", type=str, default='.*', help="experiment run name, defaul the latest run")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def _set_camera_view(env) -> None:
    eye = torch.tensor(args_cli.camera_eye, dtype=torch.float32, device=env.device)
    target = torch.tensor(args_cli.camera_target, dtype=torch.float32, device=env.device)
    if getattr(env.scene, "env_origins", None) is not None and env.scene.env_origins.numel() > 0:
        origin = env.scene.env_origins[0]
        eye = eye + origin
        target = target + origin
    env.sim.set_camera_view(eye.detach().cpu().tolist(), target.detach().cpu().tolist())


def _refresh_visualization(env, visualize_sleep: float, *, force_camera: bool = False) -> None:
    if force_camera:
        _set_camera_view(env)
    env.sim.render()
    if hasattr(simulation_app, "update"):
        simulation_app.update()
    if visualize_sleep > 0.0:
        time.sleep(visualize_sleep)


def _make_debug_actions(env, step: int, current_actions: torch.Tensor) -> torch.Tensor:
    if args_cli.action_mode == "zero":
        return torch.zeros_like(current_actions)
    if args_cli.action_mode == "random":
        if step % max(1, args_cli.action_hold_steps) == 1:
            return torch.clamp(args_cli.action_std * torch.randn_like(current_actions), -1.0, 1.0)
        return current_actions
    if args_cli.action_mode == "sine":
        phase = float(step) * args_cli.sine_frequency
        action = args_cli.action_std * torch.sin(
            phase + torch.arange(env.num_actions, device=env.device, dtype=torch.float32)
        )
        return action.unsqueeze(0).repeat(env.num_envs, 1)
    raise ValueError(f"Unsupported action_mode for debug actions: {args_cli.action_mode}")


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    # agent_cfg.load_run = "2025-08-23_00-53-24"
    # agent_cfg.load_checkpoint = "model_12000.pt"

    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.events.push_robot = None
    #env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 5
    # env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    # env_cfg.commands.ranges.lin_vel_y = (0.6, 0.6)
    # env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    livestream_mode = int(getattr(args_cli, "livestream", 0) or 0)
    env_headless = bool(args_cli.headless)
    if livestream_mode > 0:
        env_headless = False
    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, env_headless)
    print(
        f"[INFO] Play render mode: args_headless={args_cli.headless}, "
        f"env_headless={env_headless}, livestream={livestream_mode}"
    )
    visualize_mode = (not args_cli.headless) or livestream_mode > 0
    visualize_sleep = args_cli.visualize_sleep
    if visualize_sleep is None:
        visualize_sleep = 0.03 if livestream_mode > 0 else 0.0
    warmup_render_steps = args_cli.warmup_render_steps
    if warmup_render_steps is None:
        warmup_render_steps = 30 if livestream_mode > 0 else 0
    if visualize_mode:
        try:
            _set_camera_view(env)
            print(f"[INFO] Camera view set: eye={args_cli.camera_eye}, target={args_cli.camera_target}")
            for _ in range(max(0, warmup_render_steps)):
                _refresh_visualization(env, visualize_sleep, force_camera=True)
        except Exception as exc:
            print(f"[WARN] Failed to set camera view: {exc}")

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    # agent_cfg.load_run=args_cli.run
    # agent_cfg.load_checkpoint=args_cli.checkpoint
    runner = None
    policy = None
    log_dir = log_root_path
    if args_cli.no_load_runner:
        if args_cli.action_mode == "policy":
            raise ValueError("--no_load_runner requires --action_mode to be zero, random, or sine.")
        print("[INFO] Skipping runner/checkpoint/policy loading before play.")
    else:
        print(f"\n[INFO] Loading experiment from directory: {log_root_path}")
        print(f'agent_cfg.load_run:{agent_cfg.load_run}')
        print(f'agent_cfg.load_checkpoint:{agent_cfg.load_checkpoint}\n')
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        resume_path = get_checkpoint_path(log_root_path, args_cli.load_run, args_cli.checkpoint)
        log_dir = os.path.dirname(resume_path)

        # Choose runner implementation
        if args_cli.predictor:
            from rsl_rl.rsl_rl.runners import OnPolicyPredictorRegressionRunner
            runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.load(resume_path, load_optimizer=False)

        policy = runner.get_inference_policy(device=env.device)

        if args_cli.skip_export:
            print("[INFO] Skipping predictor/JIT/ONNX export before play.")
        else:
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            os.makedirs(export_model_dir, exist_ok=True)
            # Export predictor weights only when predictor-augmented runner is used
            if args_cli.predictor:
                # Save predictor as generic TorchScript module (policy-specific exporter not applicable)
                # Move to CPU before scripting/saving to avoid CUDA dependency during load.
                try:
                    orig_device = next(runner._predictor.parameters()).device
                except StopIteration:
                    orig_device = torch.device("cpu")
                runner._predictor.to("cpu").eval()
                ts_predictor = torch.jit.script(runner._predictor)
                ts_predictor.save(os.path.join(export_model_dir, "predictor.pt"))
                # Restore original device for runtime
                runner._predictor.to(orig_device)
            # Export policy in both JIT and ONNX formats
            export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(
                runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()
    # Prepare record buffers if requested
    record_action = bool(getattr(args_cli, "record_action", False))
    if record_action:
        obs_records = []        # list[np.ndarray] of shape (obs_dim,)
        action_records = []     # list[np.ndarray] of shape (act_dim,)
        obs_slice_records = []  # list[np.ndarray] of shape (21,) for obs[48:69]
    # Success / serve counters for play session
    succ_total, hit_total, serve_total = 0, 0, 0
    try:
        # per-env flag: whether current serve achieved success
        serve_success_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # per-env flag: whether current serve had any paddle hit
        serve_hit_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    except Exception:
        serve_success_flag = None
        serve_hit_flag = None
    step_count = 0
    debug_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    try:
        while simulation_app.is_running():

            with torch.inference_mode():
                if args_cli.action_mode == "policy":
                    if policy is None:
                        raise RuntimeError("Policy action mode requires a loaded runner/policy.")
                    actions = policy(obs)
                else:
                    debug_actions = _make_debug_actions(env, step_count + 1, debug_actions)
                    actions = debug_actions
                # Record inputs/outputs of policy inference for env 0
                if record_action:
                    try:
                        obs0 = obs[0].detach().to("cpu").numpy()
                        act0 = actions[0].detach().to("cpu").numpy()
                        obs_records.append(obs0)
                        action_records.append(act0)
                        # Guard slice if obs is shorter than expected
                        if obs0.shape[0] >= 69:
                            obs_slice_records.append(obs0[48:69])
                        else:
                            # Pad or slice safely
                            start = 48
                            end = min(69, obs0.shape[0])
                            pad_len = 69 - end
                            safe_slice = obs0[start:end]
                            if pad_len > 0:
                                safe_slice = np.pad(safe_slice, (0, pad_len), mode="constant")
                            obs_slice_records.append(safe_slice)
                    except Exception:
                        pass
                obs, _, _, _ = env.step(actions)
                if visualize_mode:
                    try:
                        if args_cli.keep_camera_interval > 0 and step_count % args_cli.keep_camera_interval == 0:
                            _refresh_visualization(env, visualize_sleep, force_camera=True)
                        else:
                            _refresh_visualization(env, visualize_sleep)
                    except Exception as exc:
                        print(f"[WARN] Failed to refresh GUI/WebRTC render: {exc}")
                # If predictor runner is used, update learned prediction each step for visualization/observations
                if args_cli.predictor and not args_cli.disable_predictor_update:
                    try:
                        runner._record_ball_positions()
                        runner._maybe_predict_and_update_env()
                    except Exception:
                        pass
                # Track success/serve and print periodically
                try:
                    # Update per-env success/hit flags during ongoing serve
                    if (
                        hasattr(env, "has_touch_opponent_table_just_now")
                        and hasattr(env, "has_touch_paddle")
                        and serve_success_flag is not None
                    ):
                        event_mask = (env.has_touch_opponent_table_just_now & env.has_touch_paddle)
                        serve_success_flag |= event_mask.to(serve_success_flag.device)
                    if (
                        hasattr(env, "ball_contact_rew")
                        and serve_hit_flag is not None
                    ):
                        hit_mask = (env.ball_contact_rew > 0.0)
                        serve_hit_flag |= hit_mask.to(serve_hit_flag.device)

                    # On serve boundary, aggregate and reset
                    if hasattr(env, "ball_reset_ids") and env.ball_reset_ids is not None:
                        ids = env.ball_reset_ids
                        if isinstance(ids, torch.Tensor) and ids.numel() > 0 and serve_success_flag is not None:
                            ids_dev = ids.to(serve_success_flag.device)
                            serve_total += int(ids_dev.numel())
                            succ_total += int(serve_success_flag[ids_dev].sum().item())
                            hit_total += int(serve_hit_flag[ids_dev].sum().item())
                            serve_success_flag[ids_dev] = False
                            serve_hit_flag[ids_dev] = False
                    # print(f"envheading_w {env.robot.data.heading_w}")
                except Exception:
                    pass
                step_count += 1
                if args_cli.max_play_steps > 0 and step_count >= args_cli.max_play_steps:
                    print(f"[INFO] Reached max_play_steps={args_cli.max_play_steps}.")
                    break
                if step_count % 50 == 0:
                    succ_rate = (succ_total / serve_total) if serve_total > 0 else 0.0
                    hit_rate = (hit_total / serve_total) if serve_total > 0 else 0.0
                    print(f"[Play] Success {succ_total}/{serve_total} ({succ_rate:.3f}) | Hits {hit_total}/{serve_total} ({hit_rate:.3f})")
                # print("predicted future ball pose:", env.ball_future_pose[0].cpu().numpy())
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Finalizing and saving recordings...")
    finally:
        # On exit: save recordings if requested (plotting handled by separate script)
        if record_action:
            try:
                os.makedirs(log_dir, exist_ok=True)
                # Save arrays
                obs_arr = np.asarray(obs_records) if len(obs_records) > 0 else np.empty((0,))
                act_arr = np.asarray(action_records) if len(action_records) > 0 else np.empty((0,))
                slice_arr = np.asarray(obs_slice_records) if len(obs_slice_records) > 0 else np.empty((0,))
                npz_path = os.path.join(log_dir, "play_obs_actions.npz")
                np.savez(npz_path, obs=obs_arr, actions=act_arr, obs_slice=slice_arr)
                print(f"[INFO] Saved obs/action records to: {npz_path}")
            except Exception as e:
                print(f"[WARN] Failed to save recordings: {e}")


if __name__ == "__main__":
    play()
    simulation_app.close()
