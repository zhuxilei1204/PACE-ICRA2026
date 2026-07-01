# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:    
    from legged_lab.envs.base.tt_env import TTEnv


def modify_reward_weight(env: TTEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    global_sim_step_counter = env.sim_step_counter // env.cfg.sim.decimation

    if global_sim_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        
from typing import Sequence

def modify_reward_weight_linear(env: TTEnv, env_ids: Sequence[int], term_name: str, target_weight: float, start_step: int, end_step: int):
    """
    Continuously modifies a reward term's weight using linear interpolation 
    between steps [a, b]. 

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        target_weight: The final weight value to reach at step b.
        a: The step at which interpolation should begin.
        b: The step at which interpolation should end.
    """
    global_sim_step_counter = env.sim_step_counter // env.cfg.sim.decimation
    if global_sim_step_counter <= start_step:
        return
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    current_weight = term_cfg.weight
    if global_sim_step_counter >= end_step:
        term_cfg.weight = target_weight
    else:
        dw = (target_weight - current_weight) / (end_step - global_sim_step_counter)
        term_cfg.weight = current_weight + dw
    env.reward_manager.set_term_cfg(term_name, term_cfg)


def modify_ball_ranges_piecewise_linear(env: TTEnv, env_ids: Sequence[int], phases: Sequence[dict], start_step: int = 0):
    """Linearly expands ball serve ranges across a sequence of curriculum phases."""
    if not phases:
        return

    range_keys = (
        "ball_speed_x_range",
        "ball_speed_y_range",
        "ball_speed_z_range",
        "ball_pos_y_range",
    )
    global_sim_step_counter = env.sim_step_counter // env.cfg.sim.decimation

    def apply_ranges(ranges: dict):
        for key in range_keys:
            if key in ranges:
                setattr(env.cfg.ball, key, tuple(ranges[key]))

    def lerp_ranges(start_ranges: dict, target_ranges: dict, alpha: float):
        alpha = max(0.0, min(1.0, alpha))
        out = {}
        for key in range_keys:
            if key not in target_ranges:
                continue
            start_value = start_ranges.get(key, getattr(env.cfg.ball, key))
            target_value = target_ranges[key]
            out[key] = tuple(
                float(s + (t - s) * alpha) for s, t in zip(start_value, target_value)
            )
        return out

    first_phase = phases[0]
    previous_step = int(start_step)
    previous_ranges = dict(first_phase.get("start", {}))
    if global_sim_step_counter <= previous_step:
        apply_ranges(previous_ranges)
        return

    for phase in phases:
        end_step = int(phase["end_step"])
        target_ranges = {key: phase[key] for key in range_keys if key in phase}
        if global_sim_step_counter <= end_step:
            denom = max(end_step - previous_step, 1)
            alpha = (global_sim_step_counter - previous_step) / denom
            apply_ranges(lerp_ranges(previous_ranges, target_ranges, alpha))
            return
        previous_step = end_step
        previous_ranges.update(target_ranges)

    apply_ranges(previous_ranges)


def modify_ball_ranges_by_ability(
    env: TTEnv,
    env_ids: Sequence[int],
    phases: Sequence[dict],
    min_window_steps: int = 1200,
    min_window_serves: int = 1024,
):
    """Advance A3 ball ranges only when the current policy shows enough ability.

    The state is intentionally stored on the environment object so this remains a
    plain function curriculum term and does not affect other tasks.
    """
    if not phases:
        return {}

    range_keys = (
        "ball_speed_x_range",
        "ball_speed_y_range",
        "ball_speed_z_range",
        "ball_pos_y_range",
    )

    def apply_ranges(ranges: dict):
        for key in range_keys:
            if key in ranges:
                setattr(env.cfg.ball, key, tuple(ranges[key]))

    def zero_window(stage: int, last_metrics: dict | None = None):
        return {
            "stage": int(stage),
            "steps": 0,
            "serves": 0,
            "hits": 0,
            "successes": 0,
            "fall_resets": 0,
            "episode_len_sum": 0.0,
            "episode_len_count": 0,
            "last_metrics": {} if last_metrics is None else dict(last_metrics),
        }

    def metrics_from_state(state: dict):
        serves = max(int(state["serves"]), 1)
        reset_denom = max(int(state["steps"]) * int(env.num_envs), 1)
        episode_count = int(state["episode_len_count"])
        if episode_count > 0:
            mean_episode_length = float(state["episode_len_sum"]) / float(episode_count)
        else:
            mean_episode_length = float(torch.mean(env.episode_length_buf.float()).item())
        return {
            "stage": int(state["stage"]),
            "steps": int(state["steps"]),
            "serves": int(state["serves"]),
            "hit_rate": float(state["hits"]) / float(serves),
            "success_rate": float(state["successes"]) / float(serves),
            "mean_episode_length": mean_episode_length,
            "reset_rate": float(state["fall_resets"]) / float(reset_denom),
        }

    def meets_thresholds(metrics: dict, thresholds: dict | None):
        if not thresholds:
            return False
        if metrics["steps"] < thresholds.get("min_window_steps", min_window_steps):
            return False
        if metrics["serves"] < thresholds.get("min_window_serves", min_window_serves):
            return False
        checks = (
            ("hit_rate", "min_hit_rate", lambda v, t: v >= t),
            ("hit_rate", "max_hit_rate", lambda v, t: v <= t),
            ("success_rate", "min_success_rate", lambda v, t: v >= t),
            ("success_rate", "max_success_rate", lambda v, t: v <= t),
            ("mean_episode_length", "min_mean_episode_length", lambda v, t: v >= t),
            ("mean_episode_length", "max_mean_episode_length", lambda v, t: v <= t),
            ("reset_rate", "max_reset_rate", lambda v, t: v <= t),
            ("reset_rate", "min_reset_rate", lambda v, t: v >= t),
        )
        for metric_key, threshold_key, predicate in checks:
            if threshold_key in thresholds and not predicate(metrics[metric_key], thresholds[threshold_key]):
                return False
        return True

    state = getattr(env, "_a3_ability_ball_curriculum_state", None)
    if state is None:
        state = zero_window(stage=0)

    state["steps"] += 1
    ball_reset_ids = getattr(env, "ball_reset_ids", None)
    if isinstance(ball_reset_ids, torch.Tensor):
        state["serves"] += int(ball_reset_ids.numel())

    if hasattr(env, "ball_contact_rew"):
        state["hits"] += int(torch.sum(env.ball_contact_rew > 0.0).item())
    if hasattr(env, "has_touch_opponent_table_just_now") and hasattr(env, "has_touch_paddle"):
        success = env.has_touch_opponent_table_just_now & env.has_touch_paddle
        state["successes"] += int(torch.sum(success).item())

    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor):
        time_out_buf = getattr(env, "time_out_buf", torch.zeros_like(reset_buf))
        fall_reset = reset_buf & ~time_out_buf
        state["fall_resets"] += int(torch.sum(fall_reset).item())
        done_ids = reset_buf.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0 and hasattr(env, "episode_length_buf"):
            state["episode_len_sum"] += float(torch.sum(env.episode_length_buf[done_ids].float()).item())
            state["episode_len_count"] += int(done_ids.numel())

    metrics = metrics_from_state(state)
    stage = max(0, min(int(state["stage"]), len(phases) - 1))
    changed_stage = False

    if stage > 0 and meets_thresholds(metrics, phases[stage].get("regress")):
        stage -= 1
        changed_stage = True
    elif stage < len(phases) - 1 and meets_thresholds(metrics, phases[stage].get("advance")):
        stage += 1
        changed_stage = True

    window_ready = metrics["steps"] >= min_window_steps and metrics["serves"] >= min_window_serves
    if changed_stage or window_ready:
        state = zero_window(stage=stage, last_metrics=metrics)
    else:
        state["stage"] = stage
        state["last_metrics"] = metrics

    ranges = phases[stage].get("ranges", phases[stage])
    apply_ranges(ranges)
    env._a3_ability_ball_curriculum_state = state

    log_metrics = dict(state["last_metrics"])
    log_metrics["stage"] = float(stage)
    for key in range_keys:
        if key in ranges:
            log_metrics[f"{key}_lo"] = float(ranges[key][0])
            log_metrics[f"{key}_hi"] = float(ranges[key][1])
    return log_metrics
