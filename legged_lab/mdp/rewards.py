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

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from typing import Optional

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.base.legged_env import LeggedEnv
    from legged_lab.envs.base.tt_env import TTEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def lin_vel_x_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])

def lin_vel_y_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])

def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def ang_vel_z_l2(env: TTEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: TTEnv, threshold: float, vel_ref: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    # reward *= (
    #     torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    # ) > 0.1
    reward *= (
        torch.norm(env.robot_future_vel[:, :2], dim=1)
    ) > vel_ref
    return reward

def feet_air_time_negative_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward-threshold, min=0.0)
    # no reward for zero command
    # reward *= (
    #     torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    # ) > 0.1
    return reward

def late_serve_unstable_support(
    env: TTEnv, sensor_cfg: SceneEntityCfg, min_fraction: float, max_fraction: float, force_threshold: float = 0.1
) -> torch.Tensor:
    """Counts unstable support (single-stance or both-feet-in-air) during a specified
    progress window of the ball episode.

    Returns a positive count (0 or 1) per env when the following hold:
    - The normalized ball episode progress `p = steps / max_steps` satisfies
      `min_fraction < p < max_fraction`.
    - Exactly one foot is in contact (single stance) OR both feet are in air (no contact).

    Use with a negative weight in the reward config to penalize such events.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine foot contacts from forces within recent history: max force over history > threshold
    net_forces_hist = contact_sensor.data.net_forces_w_history  # [N, T, B, 3]
    in_contact = (
        net_forces_hist[:, :, sensor_cfg.body_ids, :]  # [N, T, B, 3]
        .norm(dim=-1)                                 # [N, T, B]
        .max(dim=1)[0] > force_threshold              # [N, B]
    )
    n_contacts = torch.sum(in_contact.int(), dim=1)  # [N]

    single_stance = n_contacts == 1
    both_air = n_contacts == 0
    unstable = single_stance | both_air  # [N]

    # Window mask based on ball episode progress
    progress = env.ball_episode_length_buf.float() / float(env.max_ball_episode_length)
    late_mask = (progress > min_fraction) & (progress < max_fraction)

    # Positive count (0 or 1) to be used with a negative weight
    reward = (unstable & late_mask).float()
    return reward


def hit_unstable_support(
    env: TTEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 0.1
) -> torch.Tensor:
    """Counts unstable support exactly at hit moments.

    Returns 1 when, at the current step:
    - contact forces indicate feet state is unstable (single stance or both feet in air), and
    - the ball is in contact region with the paddle ("during hit"), approximated by ``env.ball_contact > 0``.

    Use with a negative weight in the reward config to penalize such events.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Determine foot contacts from forces within recent history: max force over history > threshold
    net_forces_hist = contact_sensor.data.net_forces_w_history  # [N, T, B, 3]
    in_contact = (
        net_forces_hist[:, :, sensor_cfg.body_ids, :]  # [N, T, B, 3]
        .norm(dim=-1)                                 # [N, T, B]
        .max(dim=1)[0] > force_threshold              # [N, B]
    )
    n_contacts = torch.sum(in_contact.int(), dim=1)  # [N]

    single_stance = n_contacts == 1
    both_air = n_contacts == 0
    unstable = single_stance | both_air  # [N] bool

    # "During hit" mask: ball currently in the paddle contact region
    # during_hit = env.ball_contact > 0.0  # [N] bool
    reward = unstable.float()*env.ball_contact_rew
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def penalty_robot_table_proximity_x(
    env: TTEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_distance: float = 0.10,
    std: float = 0.1,
) -> torch.Tensor:

    table_half_length = -1.37 - min_distance # half of table length in x-direction
    robot_pos_x = env.robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    denom=std * std + 1e-12
    penalty = torch.exp(-torch.clamp(torch.abs(robot_pos_x - table_half_length ), min=1e-6) / denom)

    return penalty

def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def paddel_too_near_humanoid(
    env: TTEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 1
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(link_pos[:, 0] - env.paddle_touch_point, dim=-1)
    return (threshold - distance).clamp(min=0)

def feet_too_high(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    excess_height = torch.clamp(feet_z - threshold, min=0.0)
    penalty = torch.sum(excess_height, dim=1) 
    return penalty

def reward_ball_z_nearnet(env: TTEnv, z_threshold: float = 0.94, x_tolerance: float = 0.03) -> torch.Tensor:
    """Reward when the ball's height exceeds ``z_threshold`` at the net plane after a paddle hit.

    Conditions (all must hold):
    - ``env.has_touch_paddle`` is True (ball has been hit).
    - Ball longitudinal position is near the net plane: ``abs(ball_x) < x_tolerance``.
    - Ball is traveling towards the opponent: ``vx > 0``.

    Returns a binary tensor of shape [N], 1.0 when conditions are met and ``z > z_threshold``, else 0.0.
    """
    # Ball state in env-local/table frame
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]

    at_net_plane = torch.abs(x) < x_tolerance
    moving_forward = vx > 0.0
    hit = env.has_touch_paddle
    high_enough = z > z_threshold

    reward = (hit & at_net_plane & moving_forward & high_enough).float()
    return reward

@torch.jit.script
def create_stance_mask(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a stance mask based on the gait phase.
    """
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(-1).repeat(1, 2)
    stance_mask = torch.where(sin_pos >= 0, 1, 0)
    stance_mask[:, 1] = 1 - stance_mask[:, 1]
    stance_mask[torch.abs(sin_pos) < 0.1] = 1

    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1
    return stance_mask, mask_2


@torch.jit.script
def compute_reward_reward_feet_contact_number(
    contacts: torch.Tensor,
    phase: torch.Tensor,
    pos_rw: float,
    neg_rw: float,
    command: torch.Tensor,
):

    stance_mask, mask_2 = create_stance_mask(phase)

    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    reward = torch.mean(reward, dim=1)
    # no reward for zero command
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def reward_feet_contact_number(
    env: LeggedEnv,
    sensor_cfg: SceneEntityCfg,
    pos_rw: float,
    neg_rw: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Calculates a reward based on the number of feet contacts aligning with the gait phase.
    Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    # print("contact", contacts.shape, contacts)
    phase = env.get_phase()
    command = env.command_generator.command[:, :2]

    return compute_reward_reward_feet_contact_number(
        contacts, phase, pos_rw, neg_rw, command
    )


@torch.jit.script
def compute_reward_foot_clearance_reward(
    com_z: torch.Tensor,
    standing_position_com_z: torch.Tensor,
    current_foot_z: torch.Tensor,
    target_height: float,
    std: float,
    tanh_mult: float,
    body_lin_vel_w: torch.Tensor,
    command: torch.Tensor,
):
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = (
        0.0626  # recorded from the default position, 0.1 compensation for walking
    )
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)
    foot_z_target_error = torch.square(
        (current_foot_z - (target_height + offset).repeat(1, 2)).clip(max=0.0)
    )
    # weighted by the velocity of the feet in the xy plane
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(body_lin_vel_w, dim=2))
    reward = foot_velocity_tanh * foot_z_target_error
    reward = torch.exp(-torch.sum(reward, dim=1) / std)
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def foot_clearance_reward(
    env: LeggedEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """
    Reward the swinging feet for clearing a specified height off the ground
    """
    com_z = env.robot.data.root_pos_w[:, 2]
    current_foot_z = env.robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # this the default standing position of the robot on the ground 
    standing_position_com_z = env.robot.data.default_root_state[:, 2]
    body_lin_vel_w = env.robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    command = env.command_generator.command[:, :2]

    return compute_reward_foot_clearance_reward(
        com_z,
        standing_position_com_z,
        current_foot_z,
        target_height,
        std,
        tanh_mult,
        body_lin_vel_w,
        command,
    )

@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0

@torch.jit.script
def compute_reward_track_foot_height(
    com_z: torch.Tensor,
    standing_position_com_z: torch.Tensor,
    phase: torch.Tensor,
    foot_z: torch.Tensor,
    standing_position_toe_roll_z: float,
    std: float,
    command: torch.Tensor,
):

    standing_height = com_z - standing_position_com_z

    offset = standing_height + standing_position_toe_roll_z

    stance_mask, mask_2 = create_stance_mask(phase)

    swing_mask = 1 - stance_mask

    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))

    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) + offset
    feet_z_value = torch.sum(filt_foot, dim=1)

    error = torch.square(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)
    # no reward for zero command
    reward *= torch.norm(command, dim=1) > 0.1
    return reward


def track_foot_height(
    env: LeggedEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    std: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """"""

    foot_z = env.robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    command = env.command_generator.command[:, :2]
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    # contacts = (
    #     contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # type: ignore
    #     .norm(dim=-1)
    #     .max(dim=1)[0]
    #     > 1.0
    # )
    com_z = env.robot.data.root_pos_w[:, 2]
    standing_position_com_z =env.robot.data.default_root_state[:, 2]
    phase = env.get_phase()

    return compute_reward_track_foot_height(
        com_z, standing_position_com_z, phase, foot_z, 0.0626, std, command
    )

@torch.compile
def bezier_curve(control_points, t):
    """
    Computes Bézier curve for given control points and parameter t.
    """
    n = len(control_points) - 1  # Degree of the Bézier curve
    dim = control_points.shape[1]  # Dimension of control points
    curve_points = torch.zeros(
        (t.shape[0], dim), dtype=control_points.dtype, device=t.device
    )

    # Calculate the Bézier curve points
    for k in range(n + 1):
        binomial_coeff = math.comb(n, k)
        bernstein_poly = binomial_coeff * (t**k) * ((1 - t) ** (n - k))
        curve_points += bernstein_poly.unsqueeze(1) * control_points[k]

    return curve_points


@torch.compile
def desired_height(phase, starting_foot):
    """
    Computes the desired heights for both legs for each environment.

    Args:
        phase (torch.Tensor): Tensor of shape (n_envs,) representing current phase values.
        starting_foot (torch.Tensor): Tensor of shape (n_envs,) with values 0 or 1 indicating which foot starts swinging first.

    Returns:
        torch.Tensor: Tensor of shape (n_envs, 2) containing desired heights for both legs.
    """
    n_envs = phase.shape[0]
    desired_heights = torch.zeros((n_envs, 2), dtype=phase.dtype, device=phase.device)

    # Step length (L) and max height (H) for the swing phase
    L = 0.5  # Step length
    H = 0.15  # Maximum height in the swing phase

    # Define control points for the swing phase Bézier curve
    control_points_swing = torch.tensor(
        [
            [0.0, 0.0],  # Start of swing phase
            [0.3 * L, 0.1 * H],  # Lift-off point
            [0.6 * L, H],  # Peak of swing
            [L, 0.0],  # Landing point
        ],
        dtype=torch.float32,
        device=phase.device,
    )

    # Loop over legs (0: left leg, 1: right leg)
    for leg in [0, 1]:
        # Determine which environments have this leg as the starting foot
        is_starting_leg = starting_foot == leg
        is_other_leg = ~is_starting_leg

        # Swing phase masks for the starting leg
        swing_mask_starting_leg = is_starting_leg & (phase >= 0.02) & (phase < 0.5)
        t_swing_starting = (phase[swing_mask_starting_leg] - 0.02) / 0.48

        # Swing phase masks for the other leg
        swing_mask_other_leg = is_other_leg & (phase >= 0.52) & (phase < 1.0)
        t_swing_other = (phase[swing_mask_other_leg] - 0.52) / 0.48

        # Combine swing masks
        swing_mask_leg = swing_mask_starting_leg | swing_mask_other_leg

        # Initialize t_swing for all environments
        t_swing = torch.zeros(n_envs, dtype=phase.dtype, device=phase.device)
        t_swing[swing_mask_starting_leg] = t_swing_starting
        t_swing[swing_mask_other_leg] = t_swing_other

        # Compute desired heights for swing phase
        if swing_mask_leg.any():
            t_swing_leg = t_swing[swing_mask_leg]
            swing_heights = bezier_curve(control_points_swing, t_swing_leg)
            desired_heights[swing_mask_leg, leg] = swing_heights[
                :, 1
            ]  # Only the y-coordinate (height)

        # Stance phase (excluding double stance): foot is on the ground (height = 0)
        # No action needed since desired_heights is initialized to zero

    # For double stance phases, both legs are already set to height = 0
    return desired_heights

def reward_contact(env: TTEnv) -> torch.Tensor:
    return env.ball_contact_rew.float()

# #(2) penalize the ball for going below a certain height.
# def penalty_ball_to_floor(env: TTEnv) -> torch.Tensor:
#     env.penalty_ball_to_floor = (env.ball_pos[:, 2] < 0.60).float()
#     return env.penalty_ball_to_floor

# # (3) encourage ball return faster x-velocity 
# def reward_vel(env: TTEnv) -> torch.Tensor:
    
#     reward_vel = (
#             env.ball_linvel[:, 0]
#             * env.has_touch_paddle.float()
#             * (env.ball_contact == 0).float()
#             * torch.logical_not(env.reward_vel_prev)
#         )
#     still_false = env.reward_vel_prev == 0    
#     env.reward_vel_prev[still_false] = reward_vel[still_false]

#     return reward_vel

# (4) reward when the ball first bounces on opponent side.
def reward_table_success(env: TTEnv) -> torch.Tensor:
    rew_table_success = (env.has_touch_paddle.float() * env.has_touch_opponent_table_just_now.float())
    return rew_table_success


def reward_opponent_table_after_paddle_hit_target(
    env: TTEnv,
    target_x: float = 1.15,
    target_y: float = 0.0,
    x_std: float = 0.7,
    y_std: float = 0.5,
) -> torch.Tensor:
    """One-shot actual opponent-table landing reward with a landing-location score."""
    hit_opponent = env.has_touch_paddle & env.has_touch_opponent_table_just_now
    x_err = torch.abs(env.ball_pos[:, 0] - target_x)
    y_err = torch.abs(env.ball_pos[:, 1] - target_y)
    x_score = torch.exp(-x_err / (x_std + 1e-12))
    y_score = torch.exp(-y_err / (y_std + 1e-12))
    reward = x_score * y_score
    reward = torch.where(hit_opponent, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_own_table_after_paddle_hit(env: TTEnv) -> torch.Tensor:
    """One-shot failure signal when a post-paddle ball lands on the own table."""
    own_table_now = getattr(env, "has_touch_own_table_just_now", torch.zeros_like(env.has_touch_paddle))
    penalty = env.has_touch_paddle & own_table_now & ~env.has_touch_opo_table_prev
    return penalty.float()


def penalty_hit_low_base_reset(env: TTEnv, min_base_z: float = 0.50) -> torch.Tensor:
    """Penalize policies that get a hit only by falling into a low-base reset."""
    reset_buf = getattr(env, "reset_buf", torch.zeros_like(env.has_touch_paddle))
    low_base = env.robot_pos[:, 2] < min_base_z
    penalty = env.has_touch_paddle & reset_buf & low_base
    return penalty.float()


def penalty_post_hit_low_base(
    env: TTEnv,
    min_base_z: float = 0.54,
    std: float = 0.06,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Dense post-hit penalty for policies that lower the base toward reset."""
    deficit = torch.clamp((min_base_z - env.robot_pos[:, 2]) / (std + 1e-12), min=0.0, max=max_penalty)
    return torch.where(env.has_touch_paddle, deficit, torch.zeros_like(deficit))


def penalty_post_hit_trajectory_excess(
    env: TTEnv,
    min_vx: float = 0.1,
    net_x: float = 0.0,
    max_z_at_net: float = 1.30,
    z_std: float = 0.35,
    vy_limit: float = 1.20,
    vy_std: float = 1.50,
    max_t_net: float = 1.40,
    max_reward_x: float = -0.95,
    z_weight: float = 0.55,
    vy_weight: float = 0.45,
) -> torch.Tensor:
    """Dense post-hit penalty for over-lifted or side-spun ball trajectories."""
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vy = env.ball_linvel[:, 1]
    vz = env.ball_linvel[:, 2]

    hit_before_table = getattr(
        env,
        "touched_paddel_no_bounce_table",
        env.has_touch_paddle
        & ~getattr(env, "has_touch_own_table_prev", torch.zeros_like(env.has_touch_paddle))
        & ~getattr(env, "has_touch_opo_table_prev", torch.zeros_like(env.has_touch_paddle)),
    )
    moving_forward = vx > min_vx

    dx_to_net = torch.clamp(net_x - x, min=0.0)
    vx_safe = torch.clamp(vx, min=min_vx)
    t_net = torch.clamp(dx_to_net / vx_safe, min=0.0, max=max_t_net)
    z_at_net = torch.where(x < net_x, z + vz * t_net - 0.5 * 9.81 * t_net * t_net, z)

    z_penalty = torch.clamp((z_at_net - max_z_at_net) / (z_std + 1e-12), min=0.0, max=1.0)
    vy_penalty = torch.clamp((torch.abs(vy) - vy_limit) / (vy_std + 1e-12), min=0.0, max=1.0)
    penalty = (z_weight * z_penalty + vy_weight * vy_penalty) / (z_weight + vy_weight + 1e-12)

    active = hit_before_table & moving_forward & (x <= max_reward_x)
    penalty = torch.where(active, penalty, torch.zeros_like(penalty))
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)

# # (5) Encourage forward ball position.
# def reward_ball_pos(env: TTEnv) -> torch.Tensor:

#     rew_table_success = reward_table_success(env)
#     rew_ball_pos = (rew_table_success * env.ball_pos[:, 0])

#     return rew_ball_pos

# # (6) Penalize ball hitting own table before opponent touch
# def penalty_table_fail(env: TTEnv) -> torch.Tensor:

#     penalty_table_fail = (env.has_first_bounce_prev * env.has_touch_own_table.float())
#     ball_x = env.ball_pos[:, 0]  # (N,)
#     mask_fail = penalty_table_fail != 0  # (N,) boolean
#     penalty_table_fail[mask_fail] += -ball_x[mask_fail] + 0.1

#     return penalty_table_fail

# def reward_paddle_ball_yz_distance_exp(env: TTEnv, std: float) -> torch.Tensor:
#     distance = torch.norm(env.ball_global_pos[:, 1:3] - env.paddle_touch_point[:, 1:3], dim=1)
#     return torch.exp(-distance / std**2) * env.has_touch_own_table

# def reward_paddle_ball_y_distance_exp(env: TTEnv, std: float) -> torch.Tensor:
#     distance = torch.abs(env.ball_global_pos[:, 1] - env.paddle_touch_point[:, 1])
#     return torch.exp(-distance / std**2)

def reward_paddle_distance_terminal(env: TTEnv, coeff: float=100.0) -> torch.Tensor:
    distance = torch.norm(env.ball_global_pos - env.paddle_touch_point, dim=1) - 0.02
    # distance = torch.norm(env.ball_global_pos[:, 1:3] - env.paddle_touch_point[:, 1:3], dim=1) - 0.02
    reward = 1 / (1 + coeff * distance**2)**2 # Tail ends at 0.25, gradient rises under 0.1
    return torch.where(env.mask_terminal, torch.zeros_like(reward), reward)

def reward_paddle_distance_terminal_weighted(env: TTEnv, coeff_x: float=100.0, coeff_y: float=100.0, coeff_z: float=100.0, weight_x: float=1.0, weight_y: float=1.0, weight_z: float=1.0) -> torch.Tensor:
    d_x = torch.abs(env.ball_global_pos[:, 0] - env.paddle_touch_point[:, 0])
    reward_x = weight_x / (1 + coeff_x * d_x**2)**2
    d_y = torch.abs(env.ball_global_pos[:, 1] - env.paddle_touch_point[:, 1])
    reward_y = weight_y / (1 + coeff_y * d_y**2)**2
    d_z = torch.abs(env.ball_global_pos[:, 2] - env.paddle_touch_point[:, 2])
    reward_z = weight_z / (1 + coeff_z * d_z**2)**2
    reward = reward_x + reward_y + reward_z
    return torch.where(env.mask_terminal, torch.zeros_like(reward), reward)

def reward_future_touch_point_target(
    env: TTEnv,
    std_ee: float = 0.5,
    threshold: float = 0.03,
) -> torch.Tensor:
    paddle_touch_point = env.paddle_touch_point - env.scene.env_origins
    distance = torch.linalg.norm(env.ball_future_pose - paddle_touch_point, dim=1)
    denom_ee = std_ee * std_ee + 1e-12
    reward_touch_point = torch.exp(-torch.clamp(distance, min=threshold) / denom_ee)
    reward_touch_point = torch.where(env.mask_invalid, torch.zeros_like(reward_touch_point), reward_touch_point)
    reward = torch.nan_to_num(reward_touch_point, nan=0.0, posinf=0.0, neginf=0.0)
    return reward

def reward_future_ee_target(
    env: TTEnv,
    std_ee: float = 0.4,
    threshold: float = 0.01,
) -> torch.Tensor:

    # dist_ee_before = torch.linalg.norm(env.pos_pred_before - env.paddle_pos, dim=1)
    # dist_ee_after = torch.linalg.norm(env.pos_pred_after - env.paddle_pos, dim=1)

    # dist_ee_before = torch.where(env.valid_before, dist_ee_before, torch.full_like(dist_ee_before, float("inf")))
    # dist_ee_after = torch.where(env.mask_after, dist_ee_after, torch.full_like(dist_ee_after, float("inf")))

    # denom_ee = std_ee * std_ee + 1e-12
    # rew_ee_before = torch.exp(-torch.clamp(dist_ee_before, min=threshold) / denom_ee)
    # rew_ee_after = torch.exp(-torch.clamp(dist_ee_after, min=threshold) / denom_ee)

    # reward_ee = torch.zeros_like(rew_ee_before)
    # reward_ee = torch.where(env.mask_before, rew_ee_before, reward_ee)
    # reward_ee = torch.where(env.mask_after, rew_ee_after, reward_ee)

    dist_ee = torch.linalg.norm(env.ball_future_pose - env.paddle_pos, dim=1)
    denom_ee = std_ee * std_ee + 1e-12
    reward_ee = torch.exp(-torch.clamp(dist_ee, min=threshold) / denom_ee)
    # mask_invalid_ee = (ball_pos[:, 0] < -1.6) | (vx > 0) | (z < 0.7) 
    reward_ee = torch.where(env.mask_invalid, torch.zeros_like(reward_ee), reward_ee)
    reward = torch.nan_to_num(reward_ee, nan=0.0, posinf=0.0, neginf=0.0)
    return reward

def paddel_ball_distance(
    env: TTEnv,
    std_ee: float = 0.3,
    threshold: float = 0.01,
) -> torch.Tensor:

    denom_ee = std_ee * std_ee + 1e-12
    rew_ee = torch.exp(-torch.clamp(env.paddel_ball_distance, min=threshold) / denom_ee)
    mask_invalid_ee = (self.ball_pos[:, 0] < -1.35) | (self.ball_pos[:, 2] < 0.75)  | env.has_touch_paddle
    reward_ee = torch.where(mask_invalid_ee, torch.zeros_like(reward_ee), reward_ee)
    reward = torch.nan_to_num(reward_ee, nan=0.0, posinf=0.0, neginf=0.0)
    return reward



def reward_future_body_target(
    env: TTEnv,
    std_ro: float = 0.5,
    threshold: float = 0.05,
) -> torch.Tensor:

    dist_ro = torch.linalg.norm(env.robot_future_pos[:, 0:2] - env.robot_pos[:, 0:2], dim=1)
    denom_ro = std_ro * std_ro + 1e-12
    rew_ro = torch.exp(-torch.clamp(dist_ro, min=threshold) / denom_ro)
    reward = torch.where(env.mask_invalid, torch.zeros_like(rew_ro), rew_ro)
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
    return reward



def reward_future_vel_target(
    env: TTEnv,
    threshold: float = 0.03,
    vel_std: float= 1.41,
) -> torch.Tensor:
    # vdiff_ro_before = torch.linalg.norm(env.vel_ro_before[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    # vdiff_ro_after = torch.linalg.norm(env.vel_ro_after[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    
    # vdiff_ro_before = torch.where(env.valid_before, vdiff_ro_before, torch.full_like(vdiff_ro_before, float(0.0)))
    # vdiff_ro_after = torch.where(env.mask_after, vdiff_ro_after, torch.full_like(vdiff_ro_after, float(0.0)))

    denom_vel=vel_std * vel_std + 1e-12
    # rew_ro_before = torch.exp(-torch.clamp(vdiff_ro_before, min=threshold) / denom_vel)
    # rew_ro_after = torch.exp(-torch.clamp(vdiff_ro_after, min=threshold) / denom_vel)
    # reward_ro = torch.zeros_like(rew_ro_before)
    # reward_ro = torch.where(env.mask_before, rew_ro_before, reward_ro)
    # reward_ro = torch.where(env.mask_after, rew_ro_after, reward_ro)

    vdiff_ro = torch.linalg.norm(env.robot_future_vel[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    reward_ro = torch.exp(-torch.clamp(vdiff_ro, min=threshold) / denom_vel)

    reward_ro = torch.where(env.mask_invalid, torch.zeros_like(reward_ro), reward_ro)
    reward = torch.nan_to_num(reward_ro, nan=0.0, posinf=0.0, neginf=0.0)
    return reward

def reward_future_landing_dis(
    env: TTEnv,
    threshold: float= 2.0,
) -> torch.Tensor:
    target_x=1.15
    target_y=0
    # Stack into 2D points
    pred_land = torch.stack([env.predict_x_land, env.predict_y_land], dim=1)
    target_land = torch.tensor([target_x, target_y], device=pred_land.device)
    # Compute 2D distance (batch-wise)
    dist_ball_land = torch.linalg.norm(pred_land - target_land, dim=1)
    # Reward = negative distance
    reward = (threshold - dist_ball_land )
    
    # reward = torch.where(env.touched_paddel_no_bounce_table, reward, torch.zeros_like(reward)) # continuous
    mask = env.ball_landing_dis_rew
    reward = torch.where(mask, reward, torch.zeros_like(reward)) # sparse
    return reward


def reward_future_opponent_landing_target(
    env: TTEnv,
    target_x: float = 1.15,
    target_y: float = 0.0,
    min_x: float = 0.0,
    std: float = 1.0,
) -> torch.Tensor:
    """Reward first post-hit predicted landing only on the opponent half.

    This is used by the A3 migration to avoid rewarding balls that are hit back
    onto the robot's own table side.
    """
    pred_land = torch.stack([env.predict_x_land, env.predict_y_land], dim=1)
    target_land = torch.tensor([target_x, target_y], device=pred_land.device, dtype=pred_land.dtype)
    dist = torch.linalg.norm(pred_land - target_land, dim=1)
    reward = torch.exp(-dist / (std * std + 1e-12))
    valid = env.ball_landing_dis_rew & torch.isfinite(dist) & (env.predict_x_land > min_x)
    reward = torch.where(valid, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_future_own_landing_after_hit(
    env: TTEnv,
    max_x: float = 0.0,
) -> torch.Tensor:
    """Flag first post-hit predicted landings on the robot's own table side."""
    valid_prediction = torch.isfinite(env.predict_x_land) & torch.isfinite(env.predict_y_land)
    own_side = env.predict_x_land <= max_x
    penalty = env.ball_landing_dis_rew & valid_prediction & own_side
    return penalty.float()


def reward_future_landing_x_progress(
    env: TTEnv,
    min_x: float = -1.5,
    target_x: float = 1.15,
    target_y: float = 0.0,
    y_std: float = 1.0,
    y_weight: float = 0.25,
) -> torch.Tensor:
    """Dense first-hit curriculum for moving predicted landing toward +x."""
    valid_prediction = torch.isfinite(env.predict_x_land) & torch.isfinite(env.predict_y_land)
    x_progress = (env.predict_x_land - min_x) / (target_x - min_x + 1e-12)
    x_progress = torch.clamp(x_progress, min=0.0, max=1.0)
    y_score = torch.exp(-torch.abs(env.predict_y_land - target_y) / (y_std + 1e-12))
    reward = x_progress * ((1.0 - y_weight) + y_weight * y_score)
    valid = env.ball_landing_dis_rew & valid_prediction
    reward = torch.where(valid, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def reward_hit_ball_velocity_net_target(
    env: TTEnv,
    vx_target: float = 3.0,
    vz_target: float = 1.5,
    z_target: float = 1.05,
    z_std: float = 0.35,
    min_vx: float = 0.1,
    max_t_net: float = 1.4,
    t_std: float = 0.7,
    vx_weight: float = 0.45,
    vz_weight: float = 0.25,
    z_weight: float = 0.20,
    t_weight: float = 0.10,
) -> torch.Tensor:
    """Stage reward for the first post-hit ball velocity and net-height potential.

    This is intended for A3 early training only: it does not replace real pass-net
    or table-success rewards.  The reward is sparse at the first hit event but
    gives a continuous score for whether that hit sends the ball forward and on a
    plausible trajectory toward the net.
    """
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vz = env.ball_linvel[:, 2]

    moving_forward = vx > min_vx
    dx_to_net = torch.clamp(0.0 - x, min=0.0)
    vx_safe = torch.clamp(vx, min=min_vx)
    t_net_raw = dx_to_net / vx_safe
    t_net = torch.clamp(t_net_raw, min=0.0, max=max_t_net)
    z_at_net = z + vz * t_net - 0.5 * 9.81 * t_net * t_net

    vx_score = torch.clamp(vx / (vx_target + 1e-12), min=0.0, max=1.0)
    vz_score = torch.clamp(vz / (vz_target + 1e-12), min=0.0, max=1.0)
    z_score = torch.exp(-torch.abs(z_at_net - z_target) / (z_std + 1e-12))
    time_score = torch.exp(-torch.clamp(t_net_raw - max_t_net, min=0.0) / (t_std + 1e-12))
    denom = vx_weight + vz_weight + z_weight + t_weight + 1e-12
    reward = (
        vx_weight * vx_score
        + vz_weight * vz_score
        + z_weight * z_score
        + t_weight * time_score
    ) / denom

    mask = env.ball_landing_dis_rew & moving_forward
    reward = torch.where(mask, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def reward_hit_net_clearance_progress(
    env: TTEnv,
    min_vx: float = 0.1,
    vx_target: float = 2.5,
    min_z: float = 0.76,
    target_z: float = 1.05,
    z_std: float = 0.45,
    max_t_net: float = 1.8,
    t_std: float = 0.8,
    vx_weight: float = 0.65,
    time_weight: float = 0.35,
) -> torch.Tensor:
    """Dense first-hit score for lifting the ball toward net clearance."""
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vz = env.ball_linvel[:, 2]

    moving_forward = vx > min_vx
    dx_to_net = torch.clamp(0.0 - x, min=0.0)
    vx_safe = torch.clamp(vx, min=min_vx)
    t_net_raw = dx_to_net / vx_safe
    t_net = torch.clamp(t_net_raw, min=0.0, max=max_t_net)
    z_at_net = z + vz * t_net - 0.5 * 9.81 * t_net * t_net

    height_deficit = torch.clamp(target_z - z_at_net, min=0.0)
    height_score = torch.exp(-height_deficit / (z_std + 1e-12))
    height_score = torch.where(z_at_net >= min_z, height_score, 0.25 * height_score)
    vx_score = torch.clamp(vx / (vx_target + 1e-12), min=0.0, max=1.0)
    time_score = torch.exp(-torch.clamp(t_net_raw - max_t_net, min=0.0) / (t_std + 1e-12))
    denom = vx_weight + time_weight + 1e-12
    forward_score = (vx_weight * vx_score + time_weight * time_score) / denom

    reward = height_score * forward_score
    mask = env.ball_landing_dis_rew & moving_forward
    reward = torch.where(mask, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def reward_post_hit_net_progress(
    env: TTEnv,
    min_vx: float = 0.1,
    vx_target: float = 3.5,
    vz_target: float = 1.6,
    x_start: float = -1.45,
    max_reward_x: float = 0.15,
    net_x: float = 0.0,
    net_z_target: float = 1.08,
    min_clear_z: float = 0.78,
    z_std: float = 0.45,
    max_t_net: float = 1.4,
    landing_min_x: float = -1.5,
    landing_target_x: float = 1.15,
    y_target: float = 0.0,
    y_std: float = 0.75,
    vy_std: float = 2.0,
    vx_weight: float = 0.25,
    vz_weight: float = 0.0,
    x_weight: float = 0.20,
    z_weight: float = 0.20,
    landing_weight: float = 0.25,
    y_weight: float = 0.10,
) -> torch.Tensor:
    """Dense A3 post-hit curriculum before the first table bounce.

    Most existing table-tennis rewards fire only on the first hit frame.  A3's
    useful hit window is narrower than T1's, so this reward keeps a short
    trajectory signal alive after paddle contact and before the first table
    contact.  It is generic, but only enabled by A3 stage configs.
    """
    x = env.ball_pos[:, 0]
    y = env.ball_pos[:, 1]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vy = env.ball_linvel[:, 1]
    vz = env.ball_linvel[:, 2]

    hit_before_table = getattr(
        env,
        "touched_paddel_no_bounce_table",
        env.has_touch_paddle
        & ~getattr(env, "has_touch_own_table_prev", torch.zeros_like(env.has_touch_paddle))
        & ~getattr(env, "has_touch_opo_table_prev", torch.zeros_like(env.has_touch_paddle)),
    )
    moving_forward = vx > min_vx

    vx_score = torch.clamp(vx / (vx_target + 1e-12), min=0.0, max=1.0)
    vz_score = torch.clamp(vz / (vz_target + 1e-12), min=0.0, max=1.0)
    x_score = torch.clamp((x - x_start) / (net_x - x_start + 1e-12), min=0.0, max=1.0)

    dx_to_net = torch.clamp(net_x - x, min=0.0)
    vx_safe = torch.clamp(vx, min=min_vx)
    t_net = torch.clamp(dx_to_net / vx_safe, min=0.0, max=max_t_net)
    z_at_net = torch.where(x < net_x, z + vz * t_net - 0.5 * 9.81 * t_net * t_net, z)
    height_score = torch.exp(-torch.abs(z_at_net - net_z_target) / (z_std + 1e-12))
    height_score = torch.where(z_at_net >= min_clear_z, height_score, 0.25 * height_score)

    pred_land_x = getattr(env, "predict_x_land", torch.full_like(x, float("nan")))
    landing_score = torch.clamp(
        (pred_land_x - landing_min_x) / (landing_target_x - landing_min_x + 1e-12),
        min=0.0,
        max=1.0,
    )
    landing_score = torch.where(torch.isfinite(pred_land_x), landing_score, torch.zeros_like(landing_score))

    y_pos_score = torch.exp(-torch.abs(y - y_target) / (y_std + 1e-12))
    y_vel_score = torch.exp(-torch.abs(vy) / (vy_std + 1e-12))
    y_score = 0.5 * (y_pos_score + y_vel_score)

    denom = vx_weight + vz_weight + x_weight + z_weight + landing_weight + y_weight + 1e-12
    reward = (
        vx_weight * vx_score
        + vz_weight * vz_score
        + x_weight * x_score
        + z_weight * height_score
        + landing_weight * landing_score
        + y_weight * y_score
    ) / denom

    active = hit_before_table & moving_forward & (x <= max_reward_x)
    reward = torch.where(active, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def reward_post_hit_ballistic_landing_target(
    env: TTEnv,
    table_z: float = 0.78,
    target_x: float = 1.15,
    target_y: float = 0.0,
    x_std: float = 0.85,
    y_std: float = 0.55,
    min_vx: float = 0.1,
    min_x: float = 0.0,
    max_x: float = 2.7,
    max_abs_y: float = 0.9,
    max_t_land: float = 1.4,
) -> torch.Tensor:
    """Dense A3 post-hit reward for estimated opponent-table landing.

    The estimate is intentionally simple and local: after paddle contact, use
    the current ball position/velocity to estimate the first crossing of table
    height.  This complements sparse actual table-success rewards without
    depending on the learned ball predictor.
    """
    x = env.ball_pos[:, 0]
    y = env.ball_pos[:, 1]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vy = env.ball_linvel[:, 1]
    vz = env.ball_linvel[:, 2]

    hit_before_table = getattr(
        env,
        "touched_paddel_no_bounce_table",
        env.has_touch_paddle
        & ~getattr(env, "has_touch_own_table_prev", torch.zeros_like(env.has_touch_paddle))
        & ~getattr(env, "has_touch_opo_table_prev", torch.zeros_like(env.has_touch_paddle)),
    )

    g = 9.81
    discriminant = vz * vz + 2.0 * g * (z - table_z)
    valid_height = discriminant >= 0.0
    sqrt_d = torch.sqrt(torch.clamp(discriminant, min=0.0))
    t_land = (vz + sqrt_d) / g
    valid_time = (t_land >= 0.0) & (t_land <= max_t_land)
    x_land = x + vx * t_land
    y_land = y + vy * t_land

    x_score = torch.exp(-torch.abs(x_land - target_x) / (x_std + 1e-12))
    y_score = torch.exp(-torch.abs(y_land - target_y) / (y_std + 1e-12))
    reward = x_score * y_score

    valid_landing = (
        (x_land >= min_x)
        & (x_land <= max_x)
        & (torch.abs(y_land) <= max_abs_y)
    )
    active = hit_before_table & (vx > min_vx) & valid_height & valid_time & valid_landing
    reward = torch.where(active, reward, torch.zeros_like(reward))
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def reward_future_pass_net(
    env: TTEnv,
    std_h : float = 0.06,
    z_target : float = 0.76 + 0.24,  # target height at net (table height + ball height above table
) -> torch.Tensor:


    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vz = env.ball_linvel[:, 2]

    # Net plane assumed at x = 0. Consider only balls moving forward (towards +x).
    moving_forward = vx > 0.0
    dx_to_net = torch.clamp(0.0 - x, min=0.0)

    # Simple flight model without drag
    # Time to net plane (avoid division by zero)
    g = 9.81
    eps = torch.tensor(1e-6, device=env.device, dtype=vx.dtype)
    vx_safe = torch.clamp(vx, min=eps)
    t_net = dx_to_net / vx_safe
    t_net = torch.clamp(t_net, min=0.0)

    # Vertical position at t_net: z + vz*t - 0.5*g*t^2
    z_at_net = z + vz * t_net - 0.5 * g * (t_net * t_net)

    # Positive reward: closer to target height is better (bell-shaped)
    height_err = torch.abs(z_at_net - z_target)
    reward = torch.exp(-height_err / (std_h + 1e-12))

    # Valid only when still moving forward to the net
    reward = torch.where(moving_forward, reward, torch.zeros_like(reward)) 
    # Apply provided sparse mask to trigger the reward once after hitting
    mask = env.ball_landing_dis_rew
    reward = torch.where(mask, reward, torch.zeros_like(reward))  # sparse
    return reward


def _normalize_w(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vec / torch.clamp(torch.linalg.norm(vec, dim=-1, keepdim=True), min=eps)


def _paddle_axis_w(env: TTEnv, local_axis: tuple[float, float, float]) -> torch.Tensor:
    paddle_quat = env.robot.data.body_quat_w[:, env.paddle_body_id, :]
    paddle_quat = paddle_quat / torch.clamp(torch.linalg.norm(paddle_quat, dim=1, keepdim=True), min=1e-6)
    axis = torch.tensor(local_axis, device=env.device, dtype=paddle_quat.dtype).unsqueeze(0).expand(env.num_envs, -1)
    return _normalize_w(math_utils.quat_apply(paddle_quat, axis))


def _a3_strike_window_score(
    env: TTEnv,
    center_t: float = 0.24,
    std_t: float = 0.18,
    min_t: float = 0.04,
    max_t: float = 0.70,
    dist_std: float | None = None,
) -> torch.Tensor:
    t_future = getattr(env, "ball_future_t", torch.zeros(env.num_envs, 1, device=env.device)).squeeze(-1)
    valid = ~getattr(env, "mask_invalid", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    valid = valid & ~getattr(env, "has_touch_paddle", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    in_time = valid & (t_future >= min_t) & (t_future <= max_t)
    time_score = torch.exp(-torch.abs(t_future - center_t) / (std_t + 1e-12))
    score = torch.where(in_time, time_score, torch.zeros_like(time_score))
    if dist_std is not None:
        paddle_pos = getattr(env, "paddle_pos", env.paddle_touch_point - env.scene.env_origins)
        dist = torch.linalg.norm(env.ball_future_pose - paddle_pos, dim=1)
        dist_score = torch.exp(-dist / (dist_std + 1e-12))
        score = score * dist_score
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def reward_strike_window_touch_point(
    env: TTEnv,
    center_t: float = 0.24,
    std_t: float = 0.18,
    min_t: float = 0.04,
    max_t: float = 0.70,
    std_ee: float = 0.38,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Reward reaching the predicted hit point only inside the useful strike window."""
    paddle_pos = getattr(env, "paddle_pos", env.paddle_touch_point - env.scene.env_origins)
    dist = torch.linalg.norm(env.ball_future_pose - paddle_pos, dim=1)
    dist_score = torch.exp(-torch.clamp(dist, min=threshold) / (std_ee + 1e-12))
    return dist_score * _a3_strike_window_score(
        env, center_t=center_t, std_t=std_t, min_t=min_t, max_t=max_t, dist_std=None
    )


def reward_paddle_normal_alignment(
    env: TTEnv,
    local_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
    center_t: float = 0.24,
    std_t: float = 0.18,
    min_t: float = 0.04,
    max_t: float = 0.70,
    dist_std: float = 0.85,
    align_power: float = 1.5,
) -> torch.Tensor:
    """Align the configured paddle face normal with the incoming ball direction."""
    normal_w = _paddle_axis_w(env, local_normal)
    incoming_dir = _normalize_w(-env.ball_linvel)
    alignment = torch.clamp(torch.sum(normal_w * incoming_dir, dim=1), min=0.0, max=1.0)
    reward = alignment.pow(align_power) * _a3_strike_window_score(
        env, center_t=center_t, std_t=std_t, min_t=min_t, max_t=max_t, dist_std=dist_std
    )
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def reward_paddle_swing_velocity_target(
    env: TTEnv,
    target_x: float = 1.15,
    target_y: float = 0.0,
    target_z: float = 1.05,
    target_speed: float = 1.20,
    min_speed: float = 0.05,
    local_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
    normal_floor: float = 0.30,
    center_t: float = 0.22,
    std_t: float = 0.16,
    min_t: float = 0.04,
    max_t: float = 0.60,
    dist_std: float = 0.70,
) -> torch.Tensor:
    """Reward paddle velocity toward the opponent side during the strike window."""
    paddle_pos = getattr(env, "paddle_pos", env.paddle_touch_point - env.scene.env_origins)
    paddle_vel = env.robot.data.body_lin_vel_w[:, env.paddle_body_id, :]
    target = torch.stack(
        [
            torch.full_like(paddle_pos[:, 0], target_x),
            torch.full_like(paddle_pos[:, 1], target_y),
            torch.full_like(paddle_pos[:, 2], target_z),
        ],
        dim=1,
    )
    swing_dir = _normalize_w(target - paddle_pos)
    forward_speed = torch.sum(paddle_vel * swing_dir, dim=1)
    speed_score = torch.clamp((forward_speed - min_speed) / (target_speed - min_speed + 1e-12), min=0.0, max=1.0)

    normal_w = _paddle_axis_w(env, local_normal)
    incoming_dir = _normalize_w(-env.ball_linvel)
    normal_score = torch.clamp(torch.sum(normal_w * incoming_dir, dim=1), min=0.0, max=1.0)
    normal_factor = normal_floor + (1.0 - normal_floor) * normal_score
    reward = speed_score * normal_factor * _a3_strike_window_score(
        env, center_t=center_t, std_t=std_t, min_t=min_t, max_t=max_t, dist_std=dist_std
    )
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_a3_forward_fall_during_strike(
    env: TTEnv,
    center_t: float = 0.24,
    std_t: float = 0.18,
    min_t: float = 0.02,
    max_t: float = 0.75,
    max_root_x: float = -1.48,
    max_forward_vx: float = 0.35,
    max_tilt: float = 0.45,
    min_base_z: float = 0.78,
    x_std: float = 0.12,
    vx_std: float = 0.45,
    tilt_std: float = 0.35,
    z_std: float = 0.12,
    x_weight: float = 0.30,
    vx_weight: float = 0.30,
    tilt_weight: float = 0.25,
    z_weight: float = 0.15,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Penalize A3 policies that reach the ball by falling toward the table."""
    asset: Articulation = env.scene["robot"]
    robot_pos = getattr(env, "robot_pos", asset.data.root_pos_w - env.scene.env_origins)

    strike_active = _a3_strike_window_score(
        env, center_t=center_t, std_t=std_t, min_t=min_t, max_t=max_t, dist_std=None
    )
    post_hit_active = getattr(env, "has_touch_paddle", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)).float()
    active = torch.clamp(torch.maximum(strike_active, post_hit_active), min=0.0, max=1.0)

    root_x = robot_pos[:, 0]
    root_z = robot_pos[:, 2]
    forward_vx = asset.data.root_lin_vel_w[:, 0]
    tilt = torch.sqrt(torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) + 1e-12)

    x_penalty = torch.clamp((root_x - max_root_x) / (x_std + 1e-12), min=0.0, max=max_penalty)
    vx_penalty = torch.clamp((forward_vx - max_forward_vx) / (vx_std + 1e-12), min=0.0, max=max_penalty)
    tilt_penalty = torch.clamp((tilt - max_tilt) / (tilt_std + 1e-12), min=0.0, max=max_penalty)
    z_penalty = torch.clamp((min_base_z - root_z) / (z_std + 1e-12), min=0.0, max=max_penalty)

    total_weight = x_weight + vx_weight + tilt_weight + z_weight + 1e-12
    penalty = (
        x_weight * x_penalty
        + vx_weight * vx_penalty
        + tilt_weight * tilt_penalty
        + z_weight * z_penalty
    ) / total_weight
    return torch.nan_to_num(active * penalty, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_scene_entity_cfg(env: TTEnv, cfg: SceneEntityCfg | None) -> SceneEntityCfg | None:
    if cfg is None:
        return None
    body_ids = getattr(cfg, "body_ids", None)
    if body_ids is None or body_ids == slice(None):
        cfg.resolve(env.scene)
    return cfg


def _a3_stability_scores(
    env: TTEnv,
    feet_sensor_cfg: SceneEntityCfg | None = None,
    bad_contact_sensor_cfg: SceneEntityCfg | None = None,
    feet_asset_cfg: SceneEntityCfg | None = None,
    min_base_z: float = 0.72,
    max_base_z: float = 1.18,
    height_std: float = 0.18,
    upright_std: float = 0.35,
    lin_vel_std: float = 1.20,
    ang_vel_std: float = 2.00,
    contact_force_threshold: float = 1.0,
    force_balance_std: float = 0.65,
    bad_contact_threshold: float = 1.0,
    bad_contact_std: float = 1.0,
    target_feet_width: float = 0.42,
    feet_width_std: float = 0.22,
) -> dict[str, torch.Tensor]:
    asset: Articulation = env.scene["robot"]
    device = env.device
    dtype = asset.data.root_pos_w.dtype
    feet_sensor_cfg = _resolve_scene_entity_cfg(env, feet_sensor_cfg)
    bad_contact_sensor_cfg = _resolve_scene_entity_cfg(env, bad_contact_sensor_cfg)
    feet_asset_cfg = _resolve_scene_entity_cfg(env, feet_asset_cfg)

    root_z = getattr(env, "robot_pos", asset.data.root_pos_w - env.scene.env_origins)[:, 2]
    low_deficit = torch.clamp(min_base_z - root_z, min=0.0)
    high_excess = torch.clamp(root_z - max_base_z, min=0.0)
    height_score = torch.exp(-(low_deficit.square() + high_excess.square()) / (height_std * height_std + 1e-12))

    upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    upright_score = torch.exp(-upright_error / (upright_std * upright_std + 1e-12))

    lin_vel_xy = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    ang_vel_xy = torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
    velocity_error = (
        lin_vel_xy.square() / (lin_vel_std * lin_vel_std + 1e-12)
        + ang_vel_xy.square() / (ang_vel_std * ang_vel_std + 1e-12)
    )
    low_velocity_score = torch.exp(-velocity_error)

    support_score = torch.ones(env.num_envs, device=device, dtype=dtype)
    if feet_sensor_cfg is None:
        feet_sensor_cfg = getattr(env, "feet_cfg", None)
    if feet_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
        foot_forces = (
            contact_sensor.data.net_forces_w_history[:, :, feet_sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        foot_contacts = foot_forces > contact_force_threshold
        contact_count = torch.sum(foot_contacts.int(), dim=1)
        if foot_forces.shape[1] >= 2:
            force_sum = foot_forces[:, 0] + foot_forces[:, 1] + 1e-12
            force_ratio_diff = torch.abs(foot_forces[:, 0] - foot_forces[:, 1]) / force_sum
            force_balance_score = torch.exp(
                -force_ratio_diff.square() / (force_balance_std * force_balance_std + 1e-12)
            )
        else:
            force_balance_score = torch.ones_like(support_score)
        support_score = torch.full_like(support_score, 0.05)
        support_score = torch.where(contact_count == 1, torch.full_like(support_score, 0.65), support_score)
        both_support_score = 0.75 + 0.25 * force_balance_score
        support_score = torch.where(contact_count >= 2, both_support_score, support_score)

    contact_clean_score = torch.ones(env.num_envs, device=device, dtype=dtype)
    if bad_contact_sensor_cfg is not None:
        contact_sensor = env.scene.sensors[bad_contact_sensor_cfg.name]
        bad_forces = (
            contact_sensor.data.net_forces_w_history[:, :, bad_contact_sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        bad_contact_count = torch.sum((bad_forces > bad_contact_threshold).float(), dim=1)
        contact_clean_score = torch.exp(-bad_contact_count / (bad_contact_std + 1e-12))

    feet_width_score = torch.ones(env.num_envs, device=device, dtype=dtype)
    if feet_asset_cfg is not None:
        feet_pos = asset.data.body_pos_w[:, feet_asset_cfg.body_ids, :2]
        if feet_pos.shape[1] >= 2:
            feet_width = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=1)
            feet_width_score = torch.exp(
                -(feet_width - target_feet_width).square() / (feet_width_std * feet_width_std + 1e-12)
            )

    return {
        "height": torch.nan_to_num(height_score, nan=0.0, posinf=0.0, neginf=0.0),
        "upright": torch.nan_to_num(upright_score, nan=0.0, posinf=0.0, neginf=0.0),
        "support": torch.nan_to_num(support_score, nan=0.0, posinf=0.0, neginf=0.0),
        "velocity": torch.nan_to_num(low_velocity_score, nan=0.0, posinf=0.0, neginf=0.0),
        "clean": torch.nan_to_num(contact_clean_score, nan=0.0, posinf=0.0, neginf=0.0),
        "feet_width": torch.nan_to_num(feet_width_score, nan=0.0, posinf=0.0, neginf=0.0),
    }


def _a3_stability_raw_score(
    env: TTEnv,
    height_weight: float = 0.28,
    upright_weight: float = 0.24,
    support_weight: float = 0.20,
    velocity_weight: float = 0.14,
    clean_weight: float = 0.09,
    feet_width_weight: float = 0.05,
    **score_kwargs,
) -> torch.Tensor:
    scores = _a3_stability_scores(env, **score_kwargs)
    weighted_scores = (
        (scores["height"], height_weight),
        (scores["upright"], upright_weight),
        (scores["support"], support_weight),
        (scores["velocity"], velocity_weight),
        (scores["clean"], clean_weight),
        (scores["feet_width"], feet_width_weight),
    )
    total_weight = sum(weight for _, weight in weighted_scores) + 1e-12
    log_score = torch.zeros(env.num_envs, device=env.device, dtype=scores["height"].dtype)
    for score, weight in weighted_scores:
        log_score = log_score + (weight / total_weight) * torch.log(torch.clamp(score, min=1e-3, max=1.0))
    return torch.exp(log_score)


def _clean_a3_stability_score_kwargs(score_kwargs: dict | None) -> dict:
    return {} if score_kwargs is None else dict(score_kwargs)


def a3_stability_gate(
    env: TTEnv,
    gate_floor: float = 0.20,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    raw_score = _a3_stability_raw_score(env, **_clean_a3_stability_score_kwargs(score_kwargs))
    gate = gate_floor + (1.0 - gate_floor) * raw_score
    return torch.clamp(torch.nan_to_num(gate, nan=gate_floor, posinf=1.0, neginf=gate_floor), min=gate_floor, max=1.0)


def reward_standing_stability(
    env: TTEnv,
    height_weight: float = 0.28,
    upright_weight: float = 0.24,
    support_weight: float = 0.20,
    velocity_weight: float = 0.14,
    clean_weight: float = 0.09,
    feet_width_weight: float = 0.05,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    score_params = _clean_a3_stability_score_kwargs(score_kwargs)
    height_weight = score_params.pop("height_weight", height_weight)
    upright_weight = score_params.pop("upright_weight", upright_weight)
    support_weight = score_params.pop("support_weight", support_weight)
    velocity_weight = score_params.pop("velocity_weight", velocity_weight)
    clean_weight = score_params.pop("clean_weight", clean_weight)
    feet_width_weight = score_params.pop("feet_width_weight", feet_width_weight)
    scores = _a3_stability_scores(env, **score_params)
    total_weight = height_weight + upright_weight + support_weight + velocity_weight + clean_weight + feet_width_weight + 1e-12
    reward = (
        height_weight * scores["height"]
        + upright_weight * scores["upright"]
        + support_weight * scores["support"]
        + velocity_weight * scores["velocity"]
        + clean_weight * scores["clean"]
        + feet_width_weight * scores["feet_width"]
    ) / total_weight
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def penalty_unstable_hit(env: TTEnv, score_kwargs: dict | None = None) -> torch.Tensor:
    raw_score = _a3_stability_raw_score(env, **_clean_a3_stability_score_kwargs(score_kwargs))
    penalty = env.ball_contact_rew.float() * (1.0 - raw_score)
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_a3_stability_gate(
    env: TTEnv,
    reward: torch.Tensor,
    gate_floor: float,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    gate = a3_stability_gate(env, gate_floor=gate_floor, score_kwargs=score_kwargs)
    return torch.nan_to_num(reward * gate, nan=0.0, posinf=0.0, neginf=0.0)


def reward_contact_stability_gated(
    env: TTEnv,
    gate_floor: float = 0.30,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    return _apply_a3_stability_gate(env, reward_contact(env), gate_floor, score_kwargs=score_kwargs)


def reward_future_touch_point_target_stability_gated(
    env: TTEnv,
    std_ee: float = 0.5,
    threshold: float = 0.03,
    gate_floor: float = 0.30,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_future_touch_point_target(env, std_ee=std_ee, threshold=threshold)
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_future_ee_target_stability_gated(
    env: TTEnv,
    std_ee: float = 0.4,
    threshold: float = 0.01,
    gate_floor: float = 0.30,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_future_ee_target(env, std_ee=std_ee, threshold=threshold)
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_strike_window_touch_point_stability_gated(
    env: TTEnv,
    center_t: float = 0.24,
    std_t: float = 0.18,
    min_t: float = 0.04,
    max_t: float = 0.70,
    std_ee: float = 0.38,
    threshold: float = 0.03,
    gate_floor: float = 0.25,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_strike_window_touch_point(
        env,
        center_t=center_t,
        std_t=std_t,
        min_t=min_t,
        max_t=max_t,
        std_ee=std_ee,
        threshold=threshold,
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_paddle_normal_alignment_stability_gated(
    env: TTEnv,
    local_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
    center_t: float = 0.24,
    std_t: float = 0.18,
    min_t: float = 0.04,
    max_t: float = 0.70,
    dist_std: float = 0.85,
    align_power: float = 1.5,
    gate_floor: float = 0.25,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_paddle_normal_alignment(
        env,
        local_normal=local_normal,
        center_t=center_t,
        std_t=std_t,
        min_t=min_t,
        max_t=max_t,
        dist_std=dist_std,
        align_power=align_power,
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_paddle_swing_velocity_target_stability_gated(
    env: TTEnv,
    target_x: float = 1.15,
    target_y: float = 0.0,
    target_z: float = 1.05,
    target_speed: float = 1.20,
    min_speed: float = 0.05,
    local_normal: tuple[float, float, float] = (0.0, 0.0, -1.0),
    normal_floor: float = 0.30,
    center_t: float = 0.22,
    std_t: float = 0.16,
    min_t: float = 0.04,
    max_t: float = 0.60,
    dist_std: float = 0.70,
    gate_floor: float = 0.25,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_paddle_swing_velocity_target(
        env,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
        target_speed=target_speed,
        min_speed=min_speed,
        local_normal=local_normal,
        normal_floor=normal_floor,
        center_t=center_t,
        std_t=std_t,
        min_t=min_t,
        max_t=max_t,
        dist_std=dist_std,
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_future_landing_x_progress_stability_gated(
    env: TTEnv,
    min_x: float = -1.5,
    target_x: float = 1.15,
    target_y: float = 0.0,
    y_std: float = 1.0,
    y_weight: float = 0.25,
    gate_floor: float = 0.15,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_future_landing_x_progress(
        env, min_x=min_x, target_x=target_x, target_y=target_y, y_std=y_std, y_weight=y_weight
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_hit_ball_velocity_net_target_stability_gated(
    env: TTEnv,
    vx_target: float = 3.0,
    vz_target: float = 1.5,
    z_target: float = 1.05,
    z_std: float = 0.35,
    min_vx: float = 0.1,
    max_t_net: float = 1.4,
    t_std: float = 0.7,
    vx_weight: float = 0.45,
    vz_weight: float = 0.25,
    z_weight: float = 0.20,
    t_weight: float = 0.10,
    gate_floor: float = 0.15,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_hit_ball_velocity_net_target(
        env,
        vx_target=vx_target,
        vz_target=vz_target,
        z_target=z_target,
        z_std=z_std,
        min_vx=min_vx,
        max_t_net=max_t_net,
        t_std=t_std,
        vx_weight=vx_weight,
        vz_weight=vz_weight,
        z_weight=z_weight,
        t_weight=t_weight,
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_hit_net_clearance_progress_stability_gated(
    env: TTEnv,
    min_vx: float = 0.1,
    vx_target: float = 2.5,
    min_z: float = 0.76,
    target_z: float = 1.05,
    z_std: float = 0.45,
    max_t_net: float = 1.8,
    t_std: float = 0.8,
    vx_weight: float = 0.65,
    time_weight: float = 0.35,
    gate_floor: float = 0.15,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_hit_net_clearance_progress(
        env,
        min_vx=min_vx,
        vx_target=vx_target,
        min_z=min_z,
        target_z=target_z,
        z_std=z_std,
        max_t_net=max_t_net,
        t_std=t_std,
        vx_weight=vx_weight,
        time_weight=time_weight,
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_post_hit_net_progress_stability_gated(
    env: TTEnv,
    gate_floor: float = 0.15,
    reward_kwargs: dict | None = None,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_post_hit_net_progress(env, **({} if reward_kwargs is None else dict(reward_kwargs)))
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_future_pass_net_stability_gated(
    env: TTEnv,
    std_h: float = 0.06,
    z_target: float = 0.76 + 0.24,
    gate_floor: float = 0.15,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_future_pass_net(env, std_h=std_h, z_target=z_target)
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)


def reward_table_success_stability_gated(
    env: TTEnv,
    gate_floor: float = 0.10,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    return _apply_a3_stability_gate(env, reward_table_success(env), gate_floor, score_kwargs=score_kwargs)


def reward_opponent_table_after_paddle_hit_target_stability_gated(
    env: TTEnv,
    target_x: float = 1.15,
    target_y: float = 0.0,
    x_std: float = 0.7,
    y_std: float = 0.5,
    gate_floor: float = 0.10,
    score_kwargs: dict | None = None,
) -> torch.Tensor:
    reward = reward_opponent_table_after_paddle_hit_target(
        env, target_x=target_x, target_y=target_y, x_std=x_std, y_std=y_std
    )
    return _apply_a3_stability_gate(env, reward, gate_floor, score_kwargs=score_kwargs)

def robot_px_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - asset.data.default_root_state[:, 0])

def robot_heading_quad(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.heading_w - target_heading)

def body_heading_quad(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_heading = yaw
    return torch.square(body_heading - target_heading)

def body_heading_exp(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_heading = yaw
    return torch.exp(-torch.abs(body_heading - target_heading) / std)

def body_pitch_exp(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_pitch = pitch
    return torch.exp(-torch.abs(body_pitch - target) / std)

def body_pitch_contact_exp(env: TTEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    body_pitch = pitch
    reward = torch.exp(-torch.abs(body_pitch - target) / std)
    return (env.ball_contact_rew > 0).float() * reward


def penalty_stand_still(
    env: TTEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 0.1, move_threshold: float = 0.1
):  
    """
    For TTEnv only.
    Penalize the robot for standing still when both feet are in contact
    and the predicted movement is below a given threshold.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Recent contact forces, only for specified body IDs
    contacts = (
        net_contact_forces[:, :, sensor_cfg.body_ids, :]  # (N, T, B, 3)
        .norm(dim=-1)                                     # (N, T, B)
        .max(dim=1)[0] > force_threshold                        # (N, B)
    )

    # True if all specified bodies are in contact
    both_feet_in_contact = torch.all(contacts, dim=1)     # (N,)

    # Compute movement magnitude (L2 norm between predicted and current position)
    position_diff = torch.norm(env.robot_future_pos - env.robot_pos, dim=-1)  # (N,)

    # Apply penalty only if robot is standing still (contact) AND not moving enough
    penalty = both_feet_in_contact & (position_diff > move_threshold)
    penalty = penalty.float()
    return penalty
