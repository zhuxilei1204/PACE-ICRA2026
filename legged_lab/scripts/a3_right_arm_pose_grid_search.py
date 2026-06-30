"""Batch-search A3 right-arm default poses for passive table-tennis hits.

This diagnostic is A3-only.  It assigns one right-arm/wrist candidate to each
parallel Isaac environment, keeps zero actions, and measures the first-hit ball
outcome.  The goal is to find default poses whose paddle geometry sends the ball
more forward/upward before spending time on long PPO runs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.scripts.a3_hit_outcome_diagnostics import _configure_env, _estimate_net_z
from legged_lab.utils import task_registry


SEARCH_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _float_list(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None or text.strip() == "":
        return default
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _make_candidates(args: argparse.Namespace) -> list[dict[str, float]]:
    values = {
        "right_shoulder_pitch_joint": _float_list(args.rsp_values, (0.1449383158874511,)),
        "right_shoulder_roll_joint": _float_list(args.rsr_values, (-0.053864232177734285,)),
        "right_shoulder_yaw_joint": _float_list(args.rsy_values, (0.004107922210693449,)),
        "right_elbow_joint": _float_list(args.re_values, (0.30, 0.41, 0.55)),
        "right_wrist_roll_joint": _float_list(args.rwr_values, (-0.50, -0.25, 0.0)),
        "right_wrist_pitch_joint": _float_list(args.rwp_values, (-1.45, -1.25, -1.05, -0.85)),
        "right_wrist_yaw_joint": _float_list(args.rwy_values, (-1.60, -1.40, -1.20)),
    }
    keys = list(values.keys())
    candidates = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(values[key] for key in keys))]
    if args.limit > 0:
        candidates = candidates[: args.limit]
    return candidates


def _apply_candidates(env, candidates: list[dict[str, float]]) -> None:
    joint_ids = {name: env.robot.joint_names.index(name) for name in SEARCH_JOINTS}
    joint_pos = env.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    for env_id, candidate in enumerate(candidates):
        for joint_name, value in candidate.items():
            joint_pos[env_id, joint_ids[joint_name]] = float(value)
    env.robot.data.default_joint_pos[:] = joint_pos
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    env.robot.set_joint_position_target(joint_pos[:, env.action_joint_ids], joint_ids=env.action_joint_ids)


def _score_row(row: dict[str, float | int | bool]) -> float:
    score = 0.0
    if row["hit"]:
        score += 1.0
    if row["hit_vx"] > 0.0:
        score += 1.0
    if row["hit_vz"] > 0.0:
        score += 0.6
    if row["actual_crossed_net"]:
        score += 1.5
    if row["actual_net_clear"]:
        score += 2.5
    if row["opponent_table_after_hit"]:
        score += 3.0
    if row["own_table_after_hit"]:
        score -= 1.0
    if row["reset_low_z"]:
        score -= 2.0
    if row["reset_x_high"] or row["reset_x_low"] or row["reset_y_high"] or row["reset_y_low"]:
        score -= 0.8
    if torch.isfinite(torch.tensor(row["actual_z_at_net"])):
        score += 0.5 * float(row["actual_z_at_net"])
    if torch.isfinite(torch.tensor(row["hit_vx"])):
        score += 0.25 * float(row["hit_vx"])
    if torch.isfinite(torch.tensor(row["hit_vz"])):
        score += 0.15 * float(row["hit_vz"])
    return score


def _write_csv(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search A3 right-arm/wrist default poses.")
    parser.add_argument("--task", type=str, default="a3_tt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env_spacing", type=float, default=2.5)
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--net_z_target", type=float, default=1.11)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of candidates for quick smoke tests.")
    parser.add_argument("--csv", type=Path, default=None)

    parser.add_argument("--base_x", type=float, default=-0.26)
    parser.add_argument("--base_y", type=float, default=0.35)
    parser.add_argument("--base_yaw", type=float, default=0.0)
    parser.add_argument("--random_ball", action="store_true")
    parser.add_argument("--ball_y", type=float, default=0.0)
    parser.add_argument("--ball_vx", type=float, default=-5.0)
    parser.add_argument("--ball_vy", type=float, default=-0.04)
    parser.add_argument("--ball_vz", type=float, default=1.5)
    parser.add_argument("--right_arm_noise", type=float, default=0.0)

    parser.add_argument("--rsp_values", type=str, default=None)
    parser.add_argument("--rsr_values", type=str, default=None)
    parser.add_argument("--rsy_values", type=str, default=None)
    parser.add_argument("--re_values", type=str, default=None)
    parser.add_argument("--rwr_values", type=str, default=None)
    parser.add_argument("--rwp_values", type=str, default=None)
    parser.add_argument("--rwy_values", type=str, default=None)

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    if args.task not in {"a3_tt", "a3_tt_eval"}:
        raise ValueError("This script is A3-only. Use --task=a3_tt or --task=a3_tt_eval.")

    candidates = _make_candidates(args)
    if not candidates:
        raise ValueError("No candidates generated.")
    args.num_envs = len(candidates)
    args.ignore_task_resets = True
    args.load_run = None
    args.checkpoint = None

    print(f"[A3 Right-Arm Grid] candidates: {len(candidates)}", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import legged_lab.envs  # noqa: F401

    env_cfg, _ = task_registry.get_cfgs(args.task)
    _configure_env(env_cfg, args, {})
    env_class = task_registry.get_task_class(args.task)
    env = env_class(env_cfg, headless=True)
    _apply_candidates(env, candidates)

    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    first_hit_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_hit_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    hit_pos = torch.full((env.num_envs, 3), float("nan"), dtype=torch.float, device=env.device)
    hit_vel = torch.full((env.num_envs, 3), float("nan"), dtype=torch.float, device=env.device)
    estimated_z_at_net = torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device)
    actual_crossed_net = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    actual_z_at_net = torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device)
    opponent_table_after_hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    own_table_after_hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_low_z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_x_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_x_high = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_y_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset_y_high = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    prev_ball_pos = env.ball.data.root_pos_w.detach().clone() - env.scene.env_origins
    try:
        step = 0
        while simulation_app.is_running() and step < args.max_steps:
            with torch.inference_mode():
                _, _, reset_buf, _ = env.step(actions)
                step += 1
                ball_pos = env.ball.data.root_pos_w.detach().clone() - env.scene.env_origins
                ball_vel = env.ball.data.root_lin_vel_w.detach().clone()

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
                if hasattr(env, "has_touch_own_table_prev"):
                    own_table_after_hit |= active_after_hit & env.has_touch_own_table_prev

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
                    reset_seen[new_reset] = True
                    reset_low_z[new_reset] = (robot_pos[:, 2] < 0.50)[new_reset]
                    reset_x_low[new_reset] = (robot_pos[:, 0] < -3.6)[new_reset]
                    reset_x_high[new_reset] = (robot_pos[:, 0] > -1.35)[new_reset]
                    reset_y_low[new_reset] = (robot_pos[:, 1] < -1.1)[new_reset]
                    reset_y_high[new_reset] = (robot_pos[:, 1] > 1.1)[new_reset]

                prev_ball_pos = ball_pos
    finally:
        rows: list[dict[str, float | int | bool]] = []
        for env_id, candidate in enumerate(candidates):
            row: dict[str, float | int | bool] = {
                "candidate_id": env_id,
                **candidate,
                "hit": bool(first_hit_seen[env_id].detach().cpu()),
                "hit_step": int(first_hit_step[env_id].detach().cpu()),
                "hit_x": float(hit_pos[env_id, 0].detach().cpu()),
                "hit_y": float(hit_pos[env_id, 1].detach().cpu()),
                "hit_z": float(hit_pos[env_id, 2].detach().cpu()),
                "hit_vx": float(hit_vel[env_id, 0].detach().cpu()),
                "hit_vy": float(hit_vel[env_id, 1].detach().cpu()),
                "hit_vz": float(hit_vel[env_id, 2].detach().cpu()),
                "estimated_z_at_net": float(estimated_z_at_net[env_id].detach().cpu()),
                "actual_crossed_net": bool(actual_crossed_net[env_id].detach().cpu()),
                "actual_z_at_net": float(actual_z_at_net[env_id].detach().cpu()),
                "actual_net_clear": bool(
                    actual_crossed_net[env_id].detach().cpu()
                    and torch.isfinite(actual_z_at_net[env_id]).detach().cpu()
                    and actual_z_at_net[env_id].detach().cpu() > args.net_z_target
                ),
                "opponent_table_after_hit": bool(opponent_table_after_hit[env_id].detach().cpu()),
                "own_table_after_hit": bool(own_table_after_hit[env_id].detach().cpu()),
                "reset_seen": bool(reset_seen[env_id].detach().cpu()),
                "reset_low_z": bool(reset_low_z[env_id].detach().cpu()),
                "reset_x_low": bool(reset_x_low[env_id].detach().cpu()),
                "reset_x_high": bool(reset_x_high[env_id].detach().cpu()),
                "reset_y_low": bool(reset_y_low[env_id].detach().cpu()),
                "reset_y_high": bool(reset_y_high[env_id].detach().cpu()),
            }
            row["score"] = _score_row(row)
            rows.append(row)

        rows.sort(key=lambda item: float(item["score"]), reverse=True)
        if args.csv is not None:
            _write_csv(args.csv, rows)
            print(f"[A3 Right-Arm Grid] CSV written to: {args.csv}", flush=True)

        print("\n[A3 Right-Arm Grid] Top candidates")
        for row in rows[: args.top_k]:
            print(
                "id={candidate_id:03d} score={score:.3f} hit={hit} vx={hit_vx:.3f} vz={hit_vz:.3f} "
                "z_net={actual_z_at_net:.3f} crossed={actual_crossed_net} clear={actual_net_clear} "
                "opp={opponent_table_after_hit} own={own_table_after_hit} "
                "re={right_elbow_joint:.3f} rwr={right_wrist_roll_joint:.3f} "
                "rwp={right_wrist_pitch_joint:.3f} rwy={right_wrist_yaw_joint:.3f}".format(**row),
                flush=True,
            )
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
