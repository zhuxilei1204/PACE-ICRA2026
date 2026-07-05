"""Probe constant A3 Stage-1 leg actions for x/y drift control."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from isaaclab.app import AppLauncher


@dataclass
class Candidate:
    name: str
    values: dict[str, float]


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


def _apply_joint_preset(env_cfg, preset: str, root_z: float | None) -> None:
    if preset == "config" and root_z is None:
        return

    from legged_lab.assets.a3 import A3_PINGPONG_READY_JOINT_POS, A3_STABLE_STANDING_JOINT_POS

    joint_pos = env_cfg.scene.robot.init_state.joint_pos.copy()
    if preset == "stable":
        joint_pos.update(A3_STABLE_STANDING_JOINT_POS)
    elif preset == "pingpong_ready":
        joint_pos.update(A3_PINGPONG_READY_JOINT_POS)
    elif preset != "config":
        raise ValueError(f"Unsupported joint preset: {preset}")

    root_pos = tuple(env_cfg.scene.robot.init_state.pos)
    if root_z is not None:
        root_pos = (root_pos[0], root_pos[1], float(root_z))

    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=env_cfg.scene.robot.init_state.replace(pos=root_pos, joint_pos=joint_pos)
    )


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

    env_cfg.robot.termination_robot_x_range = (-10.0, 10.0)
    env_cfg.robot.termination_robot_y_range = (-10.0, 10.0)
    if args.termination_min_z is not None:
        env_cfg.robot.termination_min_base_z = float(args.termination_min_z)
    _apply_joint_preset(env_cfg, args.joint_preset, args.root_z)


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
    if hasattr(env, "robot_pos"):
        return env.robot_pos
    return env.robot.data.root_link_pos_w - env.table.data.root_link_pos_w


def _flat_orientation_l2(env) -> torch.Tensor:
    if hasattr(env, "flat_orientation_l2_buf"):
        return env.flat_orientation_l2_buf
    gravity_xy = env.robot.data.projected_gravity_b[:, :2]
    return torch.sum(torch.square(gravity_xy), dim=1)


def _reset_cause(env, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return value


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
    parser.add_argument("--joint_preset", choices=["config", "stable", "pingpong_ready"], default="config")
    parser.add_argument("--root_z", type=float, default=None)
    parser.add_argument("--termination_min_z", type=float, default=None)
    parser.add_argument("--random_std", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--output_csv", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    if args.zero_only:
        repeat_zero_envs = max(1, int(args.repeat_zero_envs))
        candidates = [Candidate(f"zero_{i:03d}", {}) for i in range(repeat_zero_envs)]
    else:
        candidates = _candidate_specs(_parse_magnitudes(args.magnitudes))

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    env_cfg, _ = task_registry.get_cfgs(args.task)
    _configure_env(env_cfg, args, len(candidates))
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, headless=bool(args.headless))

    try:
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
        reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        low_z_reset_seen = torch.zeros_like(reset_seen)
        flat_reset_seen = torch.zeros_like(reset_seen)
        bad_posture_reset_seen = torch.zeros_like(reset_seen)
        timeout_seen = torch.zeros_like(reset_seen)
        first_reset = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        alive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        action_l2_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        action_max_abs = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
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
                pos = _robot_pos(env).detach()
                step_flat_l2 = _flat_orientation_l2(env).detach()
                max_flat_l2 = torch.maximum(max_flat_l2, step_flat_l2)
                final_flat_l2 = step_flat_l2
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
                "start_flat_l2": float(start_flat_l2[i]),
                "max_flat_l2": float(max_flat_l2[i]),
                "final_flat_l2": float(final_flat_l2[i]),
                "mean_action_l2": float(action_l2_sum[i] / max(args.steps, 1)),
                "max_action_abs": float(action_max_abs[i]),
                "low_z_reset_seen": bool(low_z_reset_seen[i].detach().cpu()),
                "flat_reset_seen": bool(flat_reset_seen[i].detach().cpu()),
                "bad_posture_reset_seen": bool(bad_posture_reset_seen[i].detach().cpu()),
                "timeout_seen": bool(timeout_seen[i].detach().cpu()),
            }
            rows.append(row)
            print(
                f"{candidate.name:<32} {dx:+.4f} {dy:+.4f} {dz:+.4f} {vx:+.4f} "
                f"{int(row['reset_seen']):>5} {row['first_reset']:>4} "
                f"{row['min_x']:+.3f}..{row['max_x']:+.3f} "
                f"flat_max={row['max_flat_l2']:.4f}"
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
