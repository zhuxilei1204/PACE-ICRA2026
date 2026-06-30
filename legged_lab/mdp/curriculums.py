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
