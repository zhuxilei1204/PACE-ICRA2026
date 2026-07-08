from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import legged_lab.mdp as mdp
from legged_lab.assets.a3 import A3_T2D5_CFG, A3_T2D5_PINGPANG_CFG
from legged_lab.assets.table_tennis.ball import BALL_CFG
from legged_lab.assets.table_tennis.table import TABLE_CFG
from legged_lab.envs.base.tt_env_config import (
    CurriculumCfg,
    RewardCfg,
    TTAgentCfg,
    TTEnvCfg,
)


A3_T2D5_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

A3_FEET_BODY_NAMES = ["left_ankle_roll_Link", "right_ankle_roll_Link"]
A3_ALLOWED_TASK_CONTACT_BODY_NAMES = [
    *A3_FEET_BODY_NAMES,
    "pingpang_red_Link",
    "pingpang_black_Link",
    "right_hand_pingpang_Link",
    "pingbang_ball_Link",
]
A3_UNDESIRED_CONTACT_BODY_NAMES = (
    f"^(?!({'|'.join(A3_ALLOWED_TASK_CONTACT_BODY_NAMES)})$).*"
)
A3_INITIAL_ACTION_SCALE = 0.08
A3_ACTION_SCALE_BY_JOINT = [
    0.04,  # waist_yaw_joint
    0.04,  # waist_roll_joint
    0.04,  # waist_pitch_joint
    0.00,  # head_yaw_joint
    0.00,  # head_pitch_joint
    0.02,  # left_shoulder_pitch_joint
    0.02,  # left_shoulder_roll_joint
    0.02,  # left_shoulder_yaw_joint
    0.02,  # left_elbow_joint
    0.02,  # left_wrist_roll_joint
    0.02,  # left_wrist_pitch_joint
    0.02,  # left_wrist_yaw_joint
    0.14,  # right_shoulder_pitch_joint
    0.14,  # right_shoulder_roll_joint
    0.14,  # right_shoulder_yaw_joint
    0.18,  # right_elbow_joint
    0.20,  # right_wrist_roll_joint
    0.20,  # right_wrist_pitch_joint
    0.20,  # right_wrist_yaw_joint
    0.035,  # left_hip_pitch_joint
    0.035,  # left_hip_roll_joint
    0.035,  # left_hip_yaw_joint
    0.035,  # left_knee_joint
    0.03,  # left_ankle_pitch_joint
    0.03,  # left_ankle_roll_joint
    0.035,  # right_hip_pitch_joint
    0.035,  # right_hip_roll_joint
    0.035,  # right_hip_yaw_joint
    0.035,  # right_knee_joint
    0.03,  # right_ankle_pitch_joint
    0.03,  # right_ankle_roll_joint
]
A3_INITIAL_POLICY_NOISE_STD = 0.10
A3_TRAIN_BASE_POSE_RANGE = {"x": (-0.265, -0.255), "y": (0.34, 0.36), "yaw": (-0.02, 0.02)}
A3_TRAIN_BASE_VELOCITY_RANGE = {
    "x": (-0.005, 0.005),
    "y": (-0.005, 0.005),
    "z": (-0.005, 0.005),
    "roll": (-0.005, 0.005),
    "pitch": (-0.005, 0.005),
    "yaw": (-0.01, 0.01),
}
A3_EVAL_BASE_POSE_RANGE = {"x": (-0.26, -0.25), "y": (0.34, 0.36), "yaw": (-0.02, 0.02)}
A3_EVAL_BASE_VELOCITY_RANGE = {
    "x": (-0.01, 0.01),
    "y": (-0.01, 0.01),
    "z": (-0.01, 0.01),
    "roll": (-0.01, 0.01),
    "pitch": (-0.01, 0.01),
    "yaw": (-0.02, 0.02),
}
A3_TRAIN_BALL_SPEED_X_RANGE = (-5.2, -4.8)
A3_TRAIN_BALL_SPEED_Y_RANGE = (-0.10, 0.02)
A3_TRAIN_BALL_SPEED_Z_RANGE = (1.40, 1.60)
A3_TRAIN_BALL_POS_Y_RANGE = (-0.03, 0.03)
A3_TRAIN_CONTACT_THRESHOLD = 0.07
A3_EVAL_CONTACT_THRESHOLD = 0.05
A3_TRAIN_MAX_SERVE_PER_EPISODE = 3
A3_TRAIN_LOCOMOTION_JOINT_RESET_SCALE_RANGE = (0.98, 1.02)
A3_TRAIN_MANIPULATION_JOINT_RESET_OFFSET_RANGE = (-0.01, 0.01)
A3_STAGE5_READY_ROOT_POS = (-1.6, 0.0, 1.025)
A3_STAGE5_READY_BASE_POSE_RANGE = {"x": (-0.265, -0.255), "y": (0.34, 0.36), "yaw": (-0.005, 0.005)}
A3_STAGE5_READY_EVAL_BASE_POSE_RANGE = {"x": (-0.26, -0.26), "y": (0.35, 0.35), "yaw": (0.0, 0.0)}
A3_STAGE5_READY_BASE_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
A3_STAGE5_READY_LOCOMOTION_JOINT_RESET_SCALE_RANGE = (0.995, 1.005)
A3_STAGE5_READY_MANIPULATION_JOINT_RESET_OFFSET_RANGE = (0.0, 0.0)
A3_STAGE5_READY_LOWER_BODY_JOINT_POS = {
    "left_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.24,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.60,
    "left_ankle_pitch_joint": -0.30,
    "left_ankle_roll_joint": -0.10,
    "right_hip_pitch_joint": -0.20,
    "right_hip_roll_joint": -0.24,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.60,
    "right_ankle_pitch_joint": -0.30,
    "right_ankle_roll_joint": 0.10,
}
A3_STAGE1_ROOT_POS = (-1.6, 0.0, 1.00)
A3_STAGE1_UPPER_BODY_JOINT_POS = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": -0.06,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
A3_STAGE1_STAND_ROOT_POS = (-1.6, 0.0, 1.035)
A3_STAGE1_STAND_INIT_UPPER_BODY_JOINT_POS = {
    **A3_STAGE1_UPPER_BODY_JOINT_POS,
    "waist_pitch_joint": 0.02,
}
A3_STAGE1_LOWER_BODY_JOINT_POS = {
    "left_hip_pitch_joint": -0.02,
    "left_hip_roll_joint": 0.20,
    "left_hip_yaw_joint": 0.04,
    "left_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.16,
    "left_ankle_roll_joint": -0.09,
    "right_hip_pitch_joint": -0.02,
    "right_hip_roll_joint": -0.20,
    "right_hip_yaw_joint": -0.04,
    "right_knee_joint": 0.42,
    "right_ankle_pitch_joint": -0.16,
    "right_ankle_roll_joint": 0.09,
}
A3_STAGE1_STAND_INIT_LOWER_BODY_JOINT_POS = {
    **A3_STAGE1_LOWER_BODY_JOINT_POS,
    "left_hip_pitch_joint": -0.06,
    "left_hip_roll_joint": 0.18,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.54,
    "left_ankle_pitch_joint": -0.23,
    "left_ankle_roll_joint": -0.045,
    "right_hip_pitch_joint": -0.06,
    "right_hip_roll_joint": -0.18,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.54,
    "right_ankle_pitch_joint": -0.23,
    "right_ankle_roll_joint": 0.045,
}
A3_STAGE1_STAND_RESET_LOWER_BODY_JOINT_POS = A3_STAGE1_STAND_INIT_LOWER_BODY_JOINT_POS.copy()
A3_STAGE1_STAND_LOWER_BODY_JOINT_POS = A3_STAGE1_STAND_INIT_LOWER_BODY_JOINT_POS.copy()
A3_STAGE1_STAND_TARGET_Z = 0.950
A3_STAGE1_STAND_RESET_JOINT_POS = {
    **A3_STAGE1_STAND_INIT_UPPER_BODY_JOINT_POS,
    **A3_STAGE1_STAND_RESET_LOWER_BODY_JOINT_POS,
}
A3_STAGE5B_ACTION_SCALE_BY_JOINT = A3_ACTION_SCALE_BY_JOINT.copy()
A3_STAGE5B_ACTION_SCALE_BY_JOINT[12:19] = [
    0.18,  # right_shoulder_pitch_joint
    0.18,  # right_shoulder_roll_joint
    0.18,  # right_shoulder_yaw_joint
    0.22,  # right_elbow_joint
    0.26,  # right_wrist_roll_joint
    0.26,  # right_wrist_pitch_joint
    0.26,  # right_wrist_yaw_joint
]
A3_STAGE5D_CONTACT_THRESHOLD = 0.05
A3_STAGE5D_BALL_START_RANGES = {
    "ball_speed_x_range": (-5.15, -4.85),
    "ball_speed_y_range": (-0.12, 0.04),
    "ball_speed_z_range": (1.42, 1.62),
    "ball_pos_y_range": (-0.04, 0.04),
}
A3_STAGE5D_BALL_CURRICULUM_PHASES = [
    {
        "start": A3_STAGE5D_BALL_START_RANGES,
        "end_step": 12000,
        "ball_speed_x_range": (-5.25, -4.80),
        "ball_speed_y_range": (-0.16, 0.06),
        "ball_speed_z_range": (1.40, 1.66),
        "ball_pos_y_range": (-0.05, 0.05),
    },
    {
        "end_step": 36000,
        "ball_speed_x_range": (-5.50, -4.85),
        "ball_speed_y_range": (-0.28, 0.12),
        "ball_speed_z_range": (1.40, 1.75),
        "ball_pos_y_range": (-0.07, 0.07),
    },
    {
        "end_step": 84000,
        "ball_speed_x_range": (-5.90, -4.95),
        "ball_speed_y_range": (-0.42, 0.18),
        "ball_speed_z_range": (1.42, 1.85),
        "ball_pos_y_range": (-0.09, 0.09),
    },
    {
        "end_step": 144000,
        "ball_speed_x_range": (-6.30, -5.10),
        "ball_speed_y_range": (-0.55, 0.22),
        "ball_speed_z_range": (1.45, 1.90),
        "ball_pos_y_range": (-0.10, 0.10),
    },
]
A3_STAGE5F_ACTION_SCALE_BY_JOINT = A3_STAGE5B_ACTION_SCALE_BY_JOINT.copy()
A3_STAGE5F_ACTION_SCALE_BY_JOINT[19:31] = [
    0.050,  # left_hip_pitch_joint
    0.050,  # left_hip_roll_joint
    0.040,  # left_hip_yaw_joint
    0.055,  # left_knee_joint
    0.045,  # left_ankle_pitch_joint
    0.045,  # left_ankle_roll_joint
    0.050,  # right_hip_pitch_joint
    0.050,  # right_hip_roll_joint
    0.040,  # right_hip_yaw_joint
    0.055,  # right_knee_joint
    0.045,  # right_ankle_pitch_joint
    0.045,  # right_ankle_roll_joint
]
A3_STAGE5G_FUTURE_PADDLE_X_OFFSET = 0.224
A3_STAGE5G_FUTURE_PADDLE_Y_OFFSET = -0.397
A3_STAGE5G_INVALID_ROBOT_XY = (-1.86, 0.35)
A3_STAGE5G_WIDE_CONTACT_THRESHOLD = 0.04
A3_STAGE5G_WIDE_BALL_RANGES = {
    "ball_speed_x_range": (-5.90, -4.95),
    "ball_speed_y_range": (-0.42, 0.18),
    "ball_speed_z_range": (1.42, 1.85),
    "ball_pos_y_range": (-0.09, 0.09),
}
A3_STAGE5H_PADDLE_NORMAL_AXIS = (0.0, 0.0, -1.0)
A3_STAGE1_ACTION_SCALE_BY_JOINT = [
    0.000,  # waist_yaw_joint
    0.000,  # waist_roll_joint
    0.000,  # waist_pitch_joint
    0.00,  # head_yaw_joint
    0.00,  # head_pitch_joint
    0.000,  # left_shoulder_pitch_joint
    0.000,  # left_shoulder_roll_joint
    0.000,  # left_shoulder_yaw_joint
    0.000,  # left_elbow_joint
    0.000,  # left_wrist_roll_joint
    0.000,  # left_wrist_pitch_joint
    0.000,  # left_wrist_yaw_joint
    0.000,  # right_shoulder_pitch_joint
    0.000,  # right_shoulder_roll_joint
    0.000,  # right_shoulder_yaw_joint
    0.000,  # right_elbow_joint
    0.000,  # right_wrist_roll_joint
    0.000,  # right_wrist_pitch_joint
    0.000,  # right_wrist_yaw_joint
    0.125,  # left_hip_pitch_joint
    0.105,  # left_hip_roll_joint
    0.040,  # left_hip_yaw_joint
    0.155,  # left_knee_joint
    0.125,  # left_ankle_pitch_joint
    0.092,  # left_ankle_roll_joint
    0.125,  # right_hip_pitch_joint
    0.105,  # right_hip_roll_joint
    0.040,  # right_hip_yaw_joint
    0.155,  # right_knee_joint
    0.125,  # right_ankle_pitch_joint
    0.092,  # right_ankle_roll_joint
]
A3_STAGE1_STAND_ACTION_SCALE_BY_JOINT = [
    0.025,  # waist_yaw_joint
    0.025,  # waist_roll_joint
    0.055,  # waist_pitch_joint
    0.005,  # head_yaw_joint
    0.005,  # head_pitch_joint
    0.025,  # left_shoulder_pitch_joint
    0.020,  # left_shoulder_roll_joint
    0.015,  # left_shoulder_yaw_joint
    0.025,  # left_elbow_joint
    0.008,  # left_wrist_roll_joint
    0.008,  # left_wrist_pitch_joint
    0.008,  # left_wrist_yaw_joint
    0.025,  # right_shoulder_pitch_joint
    0.020,  # right_shoulder_roll_joint
    0.015,  # right_shoulder_yaw_joint
    0.025,  # right_elbow_joint
    0.008,  # right_wrist_roll_joint
    0.008,  # right_wrist_pitch_joint
    0.008,  # right_wrist_yaw_joint
    0.240,  # left_hip_pitch_joint
    0.160,  # left_hip_roll_joint
    0.030,  # left_hip_yaw_joint
    0.320,  # left_knee_joint
    0.115,  # left_ankle_pitch_joint
    0.055,  # left_ankle_roll_joint
    0.240,  # right_hip_pitch_joint
    0.160,  # right_hip_roll_joint
    0.030,  # right_hip_yaw_joint
    0.320,  # right_knee_joint
    0.115,  # right_ankle_pitch_joint
    0.055,  # right_ankle_roll_joint
]
A3_STAGE1_BASE_POSE_RANGE = {
    "x": (0.16, 0.16),
    "y": (0.35, 0.35),
    "roll": (-0.006, 0.006),
    "pitch": (-0.006, 0.006),
    "yaw": (0.0, 0.0),
}
A3_STAGE1_BASE_VELOCITY_RANGE = {
    "x": (-0.006, 0.006),
    "y": (-0.004, 0.004),
    "z": (-0.002, 0.002),
    "roll": (-0.015, 0.015),
    "pitch": (-0.015, 0.015),
    "yaw": (-0.006, 0.006),
}
A3_STAGE1_PUSH_VELOCITY_RANGE = {
    "x": (-0.050, 0.050),
    "y": (-0.035, 0.035),
    "z": (-0.015, 0.015),
    "roll": (-0.060, 0.060),
    "pitch": (-0.060, 0.060),
    "yaw": (-0.035, 0.035),
}
A3_STAGE1_ENABLE_INTERVAL_PUSH = False
A3_STAGE1_EVAL_BASE_POSE_RANGE = {
    "x": (0.16, 0.16),
    "y": (0.35, 0.35),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
A3_STAGE1_STAND_BASE_POSE_RANGE = {
    "x": (0.16, 0.16),
    "y": (0.35, 0.35),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
A3_STAGE1_STAND_BASE_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
A3_STAGE1_LOCOMOTION_JOINT_RESET_SCALE_RANGE = (1.0, 1.0)
A3_STAGE1_MANIPULATION_JOINT_RESET_OFFSET_RANGE = (0.0, 0.0)
A3_STAGE1_BALL_RANGES = {
    "ball_speed_x_range": (0.0, 0.0),
    "ball_speed_y_range": (0.0, 0.0),
    "ball_speed_z_range": (0.0, 0.0),
    "ball_pos_y_range": (0.0, 0.0),
}
A3_STAGE1_FUTURE_PADDLE_X_OFFSET = 0.65
A3_STAGE1_FUTURE_PADDLE_Y_OFFSET = A3_STAGE5G_FUTURE_PADDLE_Y_OFFSET
A3_STAGE1_INVALID_ROBOT_XY = (-1.44, 0.35)
A3_STAGE1_TARGET_X_RANGE = (-1.45, -1.43)
A3_STAGE1_TARGET_Y_RANGE = (0.345, 0.355)
A3_STAGE1_LATERAL_TARGET_X_RANGE = (-1.440, -1.440)
A3_STAGE1_LATERAL_TARGET_Y_RANGE = (0.350, 0.350)
A3_STAGE1_TERMINATION_MIN_BASE_Z = 0.86
A3_STAGE1_TERMINATION_MAX_FLAT_ORIENTATION_L2 = 0.18
A3_STAGE1_TERMINATION_ROBOT_X_RANGE = (-4.00, 0.30)
A3_STAGE1_TERMINATION_ROBOT_Y_RANGE = (-2.50, 2.50)
A3_STAGE1_BAD_POSTURE_MIN_BASE_Z = 0.875
A3_STAGE1_BAD_POSTURE_MAX_FLAT_ORIENTATION_L2 = 0.160
A3_STAGE1_BAD_POSTURE_MAX_STEPS = 110
A3_STAGE1_DAMPING_SCALES = {
    "waist": 1.15,
    "legs": 1.65,
    "feet": 2.15,
}
A3_STAGE1_STAND_DAMPING_SCALES = {
    "waist": 1.00,
    "legs": 1.80,
    "feet": 1.25,
}
A3_STAGE1_STAND_STIFFNESS_SCALES = {
    "legs": 1.25,
    "feet": 0.90,
}
A3_STAGE1_ZERO_DELAY_ACTUATOR_GROUPS = {"waist", "legs", "feet"}
A3_STAGE5F_BALL_ABILITY_PHASES = [
    {
        "ranges": {
            "ball_speed_x_range": (-5.05, -4.75),
            "ball_speed_y_range": (-0.08, 0.03),
            "ball_speed_z_range": (1.42, 1.60),
            "ball_pos_y_range": (-0.035, 0.035),
        },
        "advance": {
            "min_window_steps": 1200,
            "min_window_serves": 2048,
            "min_mean_episode_length": 105.0,
            "min_hit_rate": 0.32,
            "max_reset_rate": 0.018,
        },
    },
    {
        "ranges": {
            "ball_speed_x_range": (-5.25, -4.80),
            "ball_speed_y_range": (-0.16, 0.06),
            "ball_speed_z_range": (1.40, 1.66),
            "ball_pos_y_range": (-0.05, 0.05),
        },
        "advance": {
            "min_window_steps": 1600,
            "min_window_serves": 2048,
            "min_mean_episode_length": 125.0,
            "min_hit_rate": 0.26,
            "max_reset_rate": 0.014,
        },
        "regress": {
            "min_window_steps": 1200,
            "min_window_serves": 1024,
            "max_mean_episode_length": 85.0,
            "max_hit_rate": 0.10,
        },
    },
    {
        "ranges": {
            "ball_speed_x_range": (-5.50, -4.85),
            "ball_speed_y_range": (-0.28, 0.12),
            "ball_speed_z_range": (1.40, 1.75),
            "ball_pos_y_range": (-0.07, 0.07),
        },
        "advance": {
            "min_window_steps": 2000,
            "min_window_serves": 2048,
            "min_mean_episode_length": 145.0,
            "min_hit_rate": 0.20,
            "max_reset_rate": 0.012,
        },
        "regress": {
            "min_window_steps": 1600,
            "min_window_serves": 1024,
            "max_mean_episode_length": 100.0,
            "max_hit_rate": 0.08,
        },
    },
    {
        "ranges": {
            "ball_speed_x_range": (-5.90, -4.95),
            "ball_speed_y_range": (-0.42, 0.18),
            "ball_speed_z_range": (1.42, 1.85),
            "ball_pos_y_range": (-0.09, 0.09),
        },
        "advance": {
            "min_window_steps": 2400,
            "min_window_serves": 2048,
            "min_mean_episode_length": 165.0,
            "min_hit_rate": 0.16,
            "min_success_rate": 0.002,
            "max_reset_rate": 0.010,
        },
        "regress": {
            "min_window_steps": 2000,
            "min_window_serves": 1024,
            "max_mean_episode_length": 115.0,
            "max_hit_rate": 0.05,
        },
    },
    {
        "ranges": {
            "ball_speed_x_range": (-6.30, -5.10),
            "ball_speed_y_range": (-0.55, 0.22),
            "ball_speed_z_range": (1.45, 1.90),
            "ball_pos_y_range": (-0.10, 0.10),
        },
        "regress": {
            "min_window_steps": 2400,
            "min_window_serves": 1024,
            "max_mean_episode_length": 130.0,
            "max_hit_rate": 0.04,
        },
    },
]


@configclass
class A3TableTennisRewardCfg(RewardCfg):
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.02)
    energy = RewTerm(func=mdp.energy, weight=-1.5e-3)
    energy_ankle = RewTerm(
        func=mdp.energy,
        weight=-2e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch_joint", ".*ankle_roll_joint"])},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.025)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-80.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=A3_UNDESIRED_CONTACT_BODY_NAMES,
            ),
            "threshold": 1.0,
        },
    )
    penalty_robot_table_proximity_x = RewTerm(
        func=mdp.penalty_robot_table_proximity_x,
        weight=-20.0,
        params={"min_distance": 0.15, "std": 0.07},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-2.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES), "threshold": 1.0},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1000.0)
    hit_unstable_support = RewTerm(
        func=mdp.hit_unstable_support,
        weight=-10,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES)},
    )
    feet_orientation_L = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_Link")},
    )
    feet_orientation_R = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_Link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    paddel_head_too_near = RewTerm(
        func=mdp.paddel_too_near_humanoid,
        weight=-100,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["head_pitch_Link"]), "threshold": 0.3},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES), "threshold": 0.2},
    )
    feet_really_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES), "threshold": 0.15},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES)},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw_joint", ".*hip_roll_joint"])},
    )
    joint_deviation_left_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"],
            )
        },
    )
    joint_deviation_left_shoulder_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_roll_joint"])},
    )
    joint_deviation_right_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    reward_contact = RewTerm(func=mdp.reward_contact, weight=110.0)
    reward_future_touch_point = RewTerm(
        func=mdp.reward_future_touch_point_target,
        weight=8.0,
        params={"std_ee": 0.5, "threshold": 0.03},
    )
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=6.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_future_dis_ro = RewTerm(
        func=mdp.reward_future_body_target,
        weight=5.0,
        params={"std_ro": 0.5, "threshold": 0.05},
    )
    reward_future_vel_base = RewTerm(
        func=mdp.reward_future_vel_target,
        weight=5.0,
        params={"vel_std": 1.2, "threshold": 0.1},
    )
    reward_future_landing_dis = RewTerm(
        func=mdp.reward_future_landing_dis,
        weight=0.0,
        params={"threshold": 3.0},
    )
    reward_future_opponent_landing = RewTerm(
        func=mdp.reward_future_opponent_landing_target,
        weight=120.0,
        params={"target_x": 1.15, "target_y": 0.0, "min_x": 0.0, "std": 1.0},
    )
    reward_future_landing_x_progress = RewTerm(
        func=mdp.reward_future_landing_x_progress,
        weight=120.0,
        params={"min_x": -3.0, "target_x": 1.15, "target_y": 0.0, "y_std": 1.0, "y_weight": 0.25},
    )
    penalty_future_own_landing = RewTerm(
        func=mdp.penalty_future_own_landing_after_hit,
        weight=-40.0,
        params={"max_x": 0.0},
    )
    penalty_actual_own_table_after_hit = RewTerm(
        func=mdp.penalty_own_table_after_paddle_hit,
        weight=-80.0,
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=160.0,
        params={
            "vx_target": 3.0,
            "vz_target": 2.0,
            "z_target": 1.10,
            "z_std": 0.35,
            "min_vx": 0.1,
            "max_t_net": 1.2,
            "t_std": 0.7,
            "vx_weight": 0.55,
            "vz_weight": 0.30,
            "z_weight": 0.10,
            "t_weight": 0.05,
        },
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress,
        weight=30.0,
        params={
            "min_vx": 0.1,
            "vx_target": 2.5,
            "min_z": 0.76,
            "target_z": 1.05,
            "z_std": 0.45,
            "max_t_net": 1.8,
            "t_std": 0.8,
            "vx_weight": 0.65,
            "time_weight": 0.35,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.4, "z_target": 0.76 + 0.35},
        weight=100.0,
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=100.0)


@configclass
class A3TableTennisEnvCfg(TTEnvCfg):
    reward = A3TableTennisRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.002
        self.sim.decimation = 10
        self.scene.height_scanner.prim_body_name = "torso_Link"
        self.scene.robot = A3_T2D5_PINGPANG_CFG
        self.scene.table = TABLE_CFG
        self.scene.ball = BALL_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.robot.num_actions = len(A3_T2D5_JOINT_NAMES)
        self.robot.num_joints = len(A3_T2D5_JOINT_NAMES)
        self.robot.action_scale = A3_ACTION_SCALE_BY_JOINT.copy()
        self.robot.terminate_contacts_body_names = ["pelvis_link", "torso_Link"]
        self.robot.feet_body_names = A3_FEET_BODY_NAMES
        self.robot.paddle_body_name = "right_hand_pingpang_Link"
        self.robot.paddle_local_offset = (0.210211399202899, 0.0320784994676765, 0.0320358706296689)
        self.robot.future_body_height = 0.90
        self.robot.future_paddle_y_offset = -0.60
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.reset_base.params["pose_range"] = A3_TRAIN_BASE_POSE_RANGE.copy()
        self.domain_rand.events.reset_base.params["velocity_range"] = A3_TRAIN_BASE_VELOCITY_RANGE.copy()
        self.ball.ball_speed_x_range = A3_TRAIN_BALL_SPEED_X_RANGE
        self.ball.ball_speed_y_range = A3_TRAIN_BALL_SPEED_Y_RANGE
        self.ball.ball_speed_z_range = A3_TRAIN_BALL_SPEED_Z_RANGE
        self.ball.ball_pos_y_range = A3_TRAIN_BALL_POS_Y_RANGE
        self.ball.contact_threshold = A3_TRAIN_CONTACT_THRESHOLD
        self.ball.max_serve_per_episode = A3_TRAIN_MAX_SERVE_PER_EPISODE
        self.domain_rand.events.reset_locomotion_joints.params["asset_cfg"].joint_names = [
            "waist_.*_joint",
            "head_.*_joint",
            ".*hip_.*_joint",
            ".*knee_joint",
            ".*ankle_.*_joint",
            "left_shoulder_.*_joint",
            "left_elbow_joint",
            "left_wrist_.*_joint",
        ]
        self.domain_rand.events.reset_locomotion_joints.params["position_range"] = (
            A3_TRAIN_LOCOMOTION_JOINT_RESET_SCALE_RANGE
        )
        self.domain_rand.events.reset_locomotion_joints.params["velocity_range"] = (0.0, 0.0)
        self.domain_rand.events.reset_manipulation_joints.params["asset_cfg"].joint_names = [
            "right_shoulder_.*_joint",
            "right_elbow_joint",
            "right_wrist_.*_joint",
        ]
        self.domain_rand.events.reset_manipulation_joints.params["position_range"] = (
            A3_TRAIN_MANIPULATION_JOINT_RESET_OFFSET_RANGE
        )
        self.domain_rand.events.reset_manipulation_joints.params["velocity_range"] = (0.0, 0.0)
        self.observations.joint_names = A3_T2D5_JOINT_NAMES
        self.actions.joint_names = A3_T2D5_JOINT_NAMES


@configclass
class A3TT_EvalEnvCfg(A3TableTennisEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.max_episode_length_s = 99999999999
        self.domain_rand.events.reset_base.params["pose_range"] = A3_EVAL_BASE_POSE_RANGE.copy()
        self.domain_rand.events.reset_base.params["velocity_range"] = A3_EVAL_BASE_VELOCITY_RANGE.copy()
        self.ball.ball_speed_x_range = (-6.5, -5.2)
        self.ball.ball_speed_y_range = (-0.6, 0.2)
        self.ball.ball_speed_z_range = (1.5, 1.9)
        self.ball.ball_pos_y_range = (-0.1, 0.1)
        self.ball.contact_threshold = A3_EVAL_CONTACT_THRESHOLD
        self.ball.max_serve_per_episode = 5


@configclass
class A3StableReturnRewardCfg(A3TableTennisRewardCfg):
    reward_contact = RewTerm(func=mdp.reward_contact, weight=120.0)
    reward_future_opponent_landing = RewTerm(
        func=mdp.reward_future_opponent_landing_target,
        weight=250.0,
        params={"target_x": 1.15, "target_y": 0.0, "min_x": 0.0, "std": 0.9},
    )
    reward_future_landing_x_progress = RewTerm(
        func=mdp.reward_future_landing_x_progress,
        weight=60.0,
        params={"min_x": -3.0, "target_x": 1.15, "target_y": 0.0, "y_std": 0.8, "y_weight": 0.35},
    )
    penalty_future_own_landing = RewTerm(
        func=mdp.penalty_future_own_landing_after_hit,
        weight=-80.0,
        params={"max_x": 0.0},
    )
    penalty_actual_own_table_after_hit = RewTerm(
        func=mdp.penalty_own_table_after_paddle_hit,
        weight=-300.0,
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=80.0,
        params={
            "vx_target": 3.0,
            "vz_target": 1.8,
            "z_target": 1.12,
            "z_std": 0.30,
            "min_vx": 0.1,
            "max_t_net": 1.2,
            "t_std": 0.7,
            "vx_weight": 0.45,
            "vz_weight": 0.20,
            "z_weight": 0.25,
            "t_weight": 0.10,
        },
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress,
        weight=20.0,
        params={
            "min_vx": 0.1,
            "vx_target": 2.5,
            "min_z": 0.76,
            "target_z": 1.05,
            "z_std": 0.45,
            "max_t_net": 1.8,
            "t_std": 0.8,
            "vx_weight": 0.65,
            "time_weight": 0.35,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.4, "z_target": 0.76 + 0.35},
        weight=120.0,
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=350.0)
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target,
        weight=250.0,
        params={"target_x": 1.15, "target_y": 0.0, "x_std": 0.7, "y_std": 0.5},
    )
    penalty_hit_low_base_reset = RewTerm(
        func=mdp.penalty_hit_low_base_reset,
        weight=-150.0,
        params={"min_base_z": 0.50},
    )


@configclass
class A3StableReturnEnvCfg(A3TableTennisEnvCfg):
    reward = A3StableReturnRewardCfg()


@configclass
class A3StableReturnEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3StableReturnRewardCfg()


@configclass
class A3Stage4bRewardCfg(A3TableTennisRewardCfg):
    reward_contact = RewTerm(func=mdp.reward_contact, weight=125.0)
    reward_future_opponent_landing = RewTerm(
        func=mdp.reward_future_opponent_landing_target,
        weight=160.0,
        params={"target_x": 1.15, "target_y": 0.0, "min_x": 0.0, "std": 0.95},
    )
    reward_future_landing_x_progress = RewTerm(
        func=mdp.reward_future_landing_x_progress,
        weight=110.0,
        params={"min_x": -3.0, "target_x": 1.15, "target_y": 0.0, "y_std": 0.9, "y_weight": 0.30},
    )
    penalty_future_own_landing = RewTerm(
        func=mdp.penalty_future_own_landing_after_hit,
        weight=-30.0,
        params={"max_x": 0.0},
    )
    penalty_actual_own_table_after_hit = RewTerm(
        func=mdp.penalty_own_table_after_paddle_hit,
        weight=-40.0,
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=190.0,
        params={
            "vx_target": 3.0,
            "vz_target": 2.0,
            "z_target": 1.10,
            "z_std": 0.35,
            "min_vx": 0.1,
            "max_t_net": 1.2,
            "t_std": 0.7,
            "vx_weight": 0.55,
            "vz_weight": 0.30,
            "z_weight": 0.10,
            "t_weight": 0.05,
        },
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress,
        weight=70.0,
        params={
            "min_vx": 0.1,
            "vx_target": 2.5,
            "min_z": 0.76,
            "target_z": 1.05,
            "z_std": 0.45,
            "max_t_net": 1.8,
            "t_std": 0.8,
            "vx_weight": 0.65,
            "time_weight": 0.35,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.4, "z_target": 0.76 + 0.35},
        weight=160.0,
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=180.0)
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target,
        weight=80.0,
        params={"target_x": 1.15, "target_y": 0.0, "x_std": 0.7, "y_std": 0.5},
    )
    penalty_hit_low_base_reset = RewTerm(
        func=mdp.penalty_hit_low_base_reset,
        weight=-20.0,
        params={"min_base_z": 0.50},
    )


@configclass
class A3Stage4bEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4bRewardCfg()


@configclass
class A3Stage4bEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4bRewardCfg()


@configclass
class A3Stage4cRewardCfg(A3Stage4bRewardCfg):
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress,
        weight=35.0,
        params={
            "min_vx": 0.1,
            "vx_target": 3.5,
            "x_start": -1.45,
            "net_x": 0.0,
            "net_z_target": 1.08,
            "min_clear_z": 0.78,
            "z_std": 0.45,
            "max_t_net": 1.4,
            "landing_min_x": -1.5,
            "landing_target_x": 1.15,
            "y_target": 0.0,
            "y_std": 0.75,
            "vy_std": 2.0,
            "vx_weight": 0.25,
            "x_weight": 0.20,
            "z_weight": 0.20,
            "landing_weight": 0.25,
            "y_weight": 0.10,
        },
    )


@configclass
class A3Stage4cEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4cRewardCfg()


@configclass
class A3Stage4cEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4cRewardCfg()


@configclass
class A3Stage4dRewardCfg(A3Stage4bRewardCfg):
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress,
        weight=18.0,
        params={
            "min_vx": 0.1,
            "vx_target": 3.5,
            "vz_target": 1.6,
            "x_start": -1.45,
            "max_reward_x": -1.05,
            "net_x": 0.0,
            "net_z_target": 1.08,
            "min_clear_z": 0.78,
            "z_std": 0.45,
            "max_t_net": 1.4,
            "landing_min_x": -1.5,
            "landing_target_x": 1.15,
            "y_target": 0.0,
            "y_std": 0.75,
            "vy_std": 2.0,
            "vx_weight": 0.15,
            "vz_weight": 0.20,
            "x_weight": 0.0,
            "z_weight": 0.30,
            "landing_weight": 0.25,
            "y_weight": 0.10,
        },
    )


@configclass
class A3Stage4dEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4dRewardCfg()


@configclass
class A3Stage4dEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4dRewardCfg()


def _apply_a3_stage5_ready_stance(env_cfg, pose_range):
    joint_pos = env_cfg.scene.robot.init_state.joint_pos.copy()
    joint_pos.update(A3_STAGE5_READY_LOWER_BODY_JOINT_POS)
    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=env_cfg.scene.robot.init_state.replace(
            pos=A3_STAGE5_READY_ROOT_POS,
            joint_pos=joint_pos,
        )
    )
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = pose_range.copy()
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = A3_STAGE5_READY_BASE_VELOCITY_RANGE.copy()
    env_cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (
        A3_STAGE5_READY_LOCOMOTION_JOINT_RESET_SCALE_RANGE
    )
    env_cfg.domain_rand.events.reset_locomotion_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (
        A3_STAGE5_READY_MANIPULATION_JOINT_RESET_OFFSET_RANGE
    )
    env_cfg.domain_rand.events.reset_manipulation_joints.params["velocity_range"] = (0.0, 0.0)


def _scale_a3_stage1_actuator_value(value, scale: float):
    if scale == 1.0:
        return value
    if isinstance(value, dict):
        return {key: float(item) * scale for key, item in value.items()}
    if isinstance(value, (float, int)):
        return float(value) * scale
    return value


def _scale_a3_stage1_actuator_damping(actuators, damping_scales=None, stiffness_scales=None):
    damping_scales = A3_STAGE1_DAMPING_SCALES if damping_scales is None else damping_scales
    stiffness_scales = {} if stiffness_scales is None else stiffness_scales
    scaled = {}
    for name, actuator in actuators.items():
        replace_kwargs = {
            "damping": _scale_a3_stage1_actuator_value(
                actuator.damping,
                damping_scales.get(name, 1.0),
            ),
            "stiffness": _scale_a3_stage1_actuator_value(
                actuator.stiffness,
                stiffness_scales.get(name, 1.0),
            ),
        }
        if name in A3_STAGE1_ZERO_DELAY_ACTUATOR_GROUPS:
            if hasattr(actuator, "min_delay"):
                replace_kwargs["min_delay"] = 0
            if hasattr(actuator, "max_delay"):
                replace_kwargs["max_delay"] = 0
        scaled[name] = actuator.replace(**replace_kwargs)
    return scaled


def _apply_a3_stage1_stance(env_cfg, pose_range):
    joint_pos = env_cfg.scene.robot.init_state.joint_pos.copy()
    joint_pos.update(A3_STAGE1_UPPER_BODY_JOINT_POS)
    joint_pos.update(A3_STAGE1_LOWER_BODY_JOINT_POS)
    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=env_cfg.scene.robot.init_state.replace(
            pos=A3_STAGE1_ROOT_POS,
            joint_pos=joint_pos,
        ),
        actuators=_scale_a3_stage1_actuator_damping(env_cfg.scene.robot.actuators),
    )
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = pose_range.copy()
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = A3_STAGE1_BASE_VELOCITY_RANGE.copy()
    env_cfg.domain_rand.events.reset_locomotion_joints.params["position_range"] = (
        A3_STAGE1_LOCOMOTION_JOINT_RESET_SCALE_RANGE
    )
    env_cfg.domain_rand.events.reset_locomotion_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.domain_rand.events.reset_manipulation_joints.params["position_range"] = (
        A3_STAGE1_MANIPULATION_JOINT_RESET_OFFSET_RANGE
    )
    env_cfg.domain_rand.events.reset_manipulation_joints.params["velocity_range"] = (0.0, 0.0)
    if env_cfg.domain_rand.events.physics_material is not None:
        env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
        env_cfg.domain_rand.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    env_cfg.domain_rand.action_delay.enable = False
    env_cfg.domain_rand.perception_delay.enable = False
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


def _apply_a3_stage1_stand_stance(env_cfg, pose_range):
    joint_pos = env_cfg.scene.robot.init_state.joint_pos.copy()
    joint_pos.update(A3_STAGE1_STAND_INIT_UPPER_BODY_JOINT_POS)
    joint_pos.update(A3_STAGE1_STAND_INIT_LOWER_BODY_JOINT_POS)
    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        init_state=env_cfg.scene.robot.init_state.replace(
            pos=A3_STAGE1_STAND_ROOT_POS,
            joint_pos=joint_pos,
        ),
        actuators=_scale_a3_stage1_actuator_damping(
            A3_T2D5_CFG.actuators,
            A3_STAGE1_STAND_DAMPING_SCALES,
            A3_STAGE1_STAND_STIFFNESS_SCALES,
        ),
    )
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = pose_range.copy()
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = A3_STAGE1_STAND_BASE_VELOCITY_RANGE.copy()
    env_cfg.domain_rand.events.reset_locomotion_joints = EventTerm(
        func=mdp.reset_joints_by_position_map,
        mode="reset",
        params={
            "joint_pos": A3_STAGE1_STAND_RESET_JOINT_POS.copy(),
            "velocity_range": (0.0, 0.0),
        },
    )
    env_cfg.domain_rand.events.reset_manipulation_joints = None


def _apply_a3_ball_ranges(env_cfg, ranges):
    for key, value in ranges.items():
        setattr(env_cfg.ball, key, tuple(value))


def _a3_stage5e_score_kwargs():
    return {
        "feet_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
        "bad_contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_UNDESIRED_CONTACT_BODY_NAMES),
        "feet_asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES),
        "min_base_z": 0.72,
        "max_base_z": 1.18,
        "height_std": 0.18,
        "upright_std": 0.35,
        "lin_vel_std": 1.20,
        "ang_vel_std": 2.00,
        "contact_force_threshold": 1.0,
        "force_balance_std": 0.65,
        "bad_contact_threshold": 1.0,
        "bad_contact_std": 1.0,
        "target_feet_width": 0.42,
        "feet_width_std": 0.22,
        "height_weight": 0.28,
        "upright_weight": 0.24,
        "support_weight": 0.20,
        "velocity_weight": 0.14,
        "clean_weight": 0.09,
        "feet_width_weight": 0.05,
    }


def _a3_stage5e_stability_params(gate_floor: float | None = None):
    params = {"score_kwargs": _a3_stage5e_score_kwargs()}
    if gate_floor is not None:
        params["gate_floor"] = gate_floor
    return params


def _a3_stage5e_gated_params(gate_floor: float, **reward_params):
    params = _a3_stage5e_stability_params(gate_floor)
    params.update(reward_params)
    return params


def _a3_stage5e_post_hit_params(gate_floor: float, **reward_kwargs):
    params = _a3_stage5e_stability_params(gate_floor)
    params["reward_kwargs"] = reward_kwargs
    return params


def _a3_stage5f_score_kwargs():
    params = _a3_stage5e_score_kwargs()
    params.update(
        {
            "height_weight": 0.32,
            "upright_weight": 0.30,
            "support_weight": 0.23,
            "velocity_weight": 0.00,
            "clean_weight": 0.10,
            "feet_width_weight": 0.05,
            "min_base_z": 0.76,
            "height_std": 0.16,
            "upright_std": 0.30,
            "target_feet_width": 0.44,
            "feet_width_std": 0.20,
        }
    )
    return params


def _a3_stage5f_stability_params(gate_floor: float | None = None):
    params = {"score_kwargs": _a3_stage5f_score_kwargs()}
    if gate_floor is not None:
        params["gate_floor"] = gate_floor
    return params


def _a3_stage5f_gated_params(gate_floor: float, **reward_params):
    params = _a3_stage5f_stability_params(gate_floor)
    params.update(reward_params)
    return params


def _a3_stage5f_post_hit_params(gate_floor: float, **reward_kwargs):
    params = _a3_stage5f_stability_params(gate_floor)
    params["reward_kwargs"] = reward_kwargs
    return params


def _a3_stage5i_score_kwargs():
    params = _a3_stage5f_score_kwargs()
    params.update(
        {
            "height_weight": 0.28,
            "upright_weight": 0.28,
            "support_weight": 0.22,
            "velocity_weight": 0.17,
            "clean_weight": 0.08,
            "feet_width_weight": 0.05,
            "min_base_z": 0.80,
            "height_std": 0.14,
            "upright_std": 0.24,
            "lin_vel_std": 0.65,
            "ang_vel_std": 1.20,
            "force_balance_std": 0.50,
        }
    )
    return params


def _a3_stage5i_stability_params(gate_floor: float | None = None):
    params = {"score_kwargs": _a3_stage5i_score_kwargs()}
    if gate_floor is not None:
        params["gate_floor"] = gate_floor
    return params


def _a3_stage5i_gated_params(gate_floor: float, **reward_params):
    params = _a3_stage5i_stability_params(gate_floor)
    params.update(reward_params)
    return params


def _a3_stage5i_post_hit_params(gate_floor: float, **reward_kwargs):
    params = _a3_stage5i_stability_params(gate_floor)
    params["reward_kwargs"] = reward_kwargs
    return params


def _a3_stage1_score_kwargs():
    params = _a3_stage5i_score_kwargs()
    params.update(
        {
            "height_weight": 0.50,
            "upright_weight": 0.34,
            "support_weight": 0.08,
            "velocity_weight": 0.03,
            "clean_weight": 0.00,
            "feet_width_weight": 0.05,
            "min_base_z": 0.94,
            "max_base_z": 1.05,
            "height_std": 0.035,
            "upright_std": 0.075,
            "lin_vel_std": 0.34,
            "ang_vel_std": 0.60,
            "force_balance_std": 0.44,
            "bad_contact_std": 0.14,
            "target_feet_width": 0.44,
            "feet_width_std": 0.18,
        }
    )
    return params


def _a3_stage1_stability_params():
    return {"score_kwargs": _a3_stage1_score_kwargs()}


def _a3_stage1_stand_score_kwargs():
    params = _a3_stage1_score_kwargs()
    params.update(
        {
            "height_weight": 0.76,
            "upright_weight": 0.18,
            "support_weight": 0.04,
            "velocity_weight": 0.01,
            "clean_weight": 0.00,
            "feet_width_weight": 0.01,
            "min_base_z": 0.950,
            "max_base_z": 1.045,
            "height_std": 0.040,
            "upright_std": 0.090,
            "lin_vel_std": 0.28,
            "ang_vel_std": 0.50,
        }
    )
    return params


def _a3_stage1_stand_stability_params():
    return {"score_kwargs": _a3_stage1_stand_score_kwargs()}


@configclass
class A3Stage1BalanceMoveRewardCfg(A3TableTennisRewardCfg):
    reward_alive = RewTerm(
        func=mdp.reward_alive_height_gated,
        weight=2.0,
        params={"min_z": 0.905, "transition": 0.060, "floor": 0.05},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-6000.0)
    lin_vel_x_l2 = RewTerm(func=mdp.lin_vel_x_l2, weight=-2.50)
    lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=-2.00)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-280.0)
    reward_base_height_target = RewTerm(
        func=mdp.reward_robot_base_height_target_stability_gated,
        weight=65.0,
        params={"target_z": 0.965, "std": 0.040, "gate_floor": 0.05, "score_kwargs": _a3_stage1_score_kwargs()},
    )
    penalty_base_height_target = RewTerm(
        func=mdp.penalty_robot_base_height_target_l2,
        weight=-220.0,
        params={"target_z": 0.965, "deadband": 0.010, "std": 0.036, "max_penalty": 14.0},
    )
    reward_base_height_hold = RewTerm(
        func=mdp.reward_robot_base_height_target,
        weight=165.0,
        params={"target_z": 0.965, "std": 0.035},
    )
    reward_base_height_recovery = RewTerm(
        func=mdp.reward_robot_base_height_target,
        weight=55.0,
        params={"target_z": 0.965, "std": 0.075},
    )
    reward_upright_recovery = RewTerm(
        func=mdp.reward_upright_exp,
        weight=60.0,
        params={"std": 0.20},
    )
    reward_base_height_recovery_rate = RewTerm(
        func=mdp.reward_base_height_recovery_rate,
        weight=260.0,
        params={"target_z": 0.965, "deadband": 0.012, "std": 0.045, "max_rate": 1.0},
    )
    reward_upright_recovery_rate = RewTerm(
        func=mdp.reward_upright_recovery_rate,
        weight=260.0,
        params={"deadband": 0.018, "std": 0.050, "max_rate": 1.0},
    )
    reward_xy_anchor = RewTerm(
        func=mdp.reward_robot_xy_target_stability_gated,
        weight=0.8,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "std": 0.20,
            "gate_floor": 0.0,
            "score_kwargs": _a3_stage1_score_kwargs(),
        },
    )
    penalty_stage1_xy_drift = RewTerm(
        func=mdp.penalty_robot_xy_drift,
        weight=-3.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.06,
            "y_margin": 0.06,
            "std": 0.14,
            "max_penalty": 10.0,
        },
    )
    penalty_stage1_forward_x = RewTerm(
        func=mdp.penalty_robot_x_upper_bound,
        weight=-12.0,
        params={"max_x": -1.32, "std": 0.14, "max_penalty": 6.0},
    )
    penalty_stage1_forward_x_velocity = RewTerm(
        func=mdp.penalty_robot_forward_x_velocity,
        weight=-3.0,
        params={"max_vx": 0.015, "std": 0.12, "max_penalty": 4.0},
    )
    penalty_stage1_forward_x_velocity_bound = RewTerm(
        func=mdp.penalty_robot_forward_x_velocity_after_bound,
        weight=-4.0,
        params={"min_x": -1.38, "max_vx": 0.0, "x_std": 0.12, "vx_std": 0.12, "max_penalty": 6.0},
    )
    reward_stage1_xy_return_velocity = RewTerm(
        func=mdp.reward_robot_axis_velocity_towards_target_stability_gated,
        weight=1.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.05,
            "y_margin": 0.05,
            "max_x_speed": 0.06,
            "max_y_speed": 0.04,
            "x_weight": 0.80,
            "y_weight": 0.20,
            "gate_floor": 0.0,
            "score_kwargs": _a3_stage1_score_kwargs(),
        },
    )
    reward_stage1_xy_return_velocity_raw = RewTerm(
        func=mdp.reward_robot_axis_velocity_towards_target,
        weight=0.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.020,
            "y_margin": 0.020,
            "max_x_speed": 0.18,
            "max_y_speed": 0.06,
            "x_weight": 0.94,
            "y_weight": 0.06,
        },
    )
    penalty_stage1_xy_away_velocity = RewTerm(
        func=mdp.penalty_robot_axis_velocity_away_from_target,
        weight=-2.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.04,
            "y_margin": 0.04,
            "x_std": 0.14,
            "y_std": 0.14,
            "x_weight": 0.80,
            "y_weight": 0.20,
            "max_penalty": 8.0,
        },
    )
    penalty_height_margin = RewTerm(
        func=mdp.penalty_robot_low_base_height,
        weight=-800.0,
        params={"min_z": 0.955, "std": 0.035, "max_penalty": 22.0},
    )
    penalty_low_base_height_barrier = RewTerm(
        func=mdp.penalty_robot_low_base_height_barrier,
        weight=-1200.0,
        params={
            "soft_min_z": 0.950,
            "hard_min_z": A3_STAGE1_TERMINATION_MIN_BASE_Z,
            "power": 2.0,
            "max_penalty": 3.5,
        },
    )
    penalty_low_base_height = RewTerm(
        func=mdp.penalty_robot_low_base_height,
        weight=-1200.0,
        params={"min_z": 0.895, "std": 0.022, "max_penalty": 24.0},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-3.00)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-0.10)
    penalty_flat_orientation_margin = RewTerm(
        func=mdp.penalty_flat_orientation_margin,
        weight=-650.0,
        params={"max_flat_l2": 0.030, "std": 0.040, "max_penalty": 16.0},
    )
    penalty_stage1_bad_posture = RewTerm(
        func=mdp.penalty_stage1_bad_posture,
        weight=-560.0,
        params={
            "min_base_z": 0.935,
            "max_flat_l2": 0.040,
            "height_std": 0.038,
            "flat_std": 0.040,
            "max_penalty": 24.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.25)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-1.50)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-3)
    energy_ankle = RewTerm(
        func=mdp.energy,
        weight=-2.5e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch_joint", ".*ankle_roll_joint"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=A3_UNDESIRED_CONTACT_BODY_NAMES,
            ),
            "threshold": 1.0,
        },
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES), "threshold": 1.0},
    )
    reward_standing_stability = RewTerm(
        func=mdp.reward_standing_stability_height_gated,
        weight=165.0,
        params={"min_z": 0.925, "transition": 0.045, "floor": 0.05, **_a3_stage1_stability_params()},
    )
    reward_future_dis_ro = RewTerm(
        func=mdp.reward_future_body_target,
        weight=0.0,
        params={"std_ro": 1.10, "threshold": 0.04},
    )
    reward_future_vel_base = RewTerm(
        func=mdp.reward_future_vel_target,
        weight=0.0,
        params={"vel_std": 2.50, "threshold": 0.05},
    )
    feet_orientation_L = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_Link")},
    )
    feet_orientation_R = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_Link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-25.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-5.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.80,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw_joint", ".*hip_roll_joint"])},
    )
    joint_deviation_leg_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*hip_pitch_joint", ".*knee_joint", ".*ankle_pitch_joint"],
            )
        },
    )
    joint_deviation_stage1_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.00,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*hip_.*_joint", ".*knee_joint", ".*ankle_.*_joint"],
            )
        },
    )
    joint_deviation_left_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.45,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"],
            )
        },
    )
    joint_deviation_right_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.00,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )

    hit_unstable_support = None
    reward_contact = None
    reward_future_touch_point = None
    reward_future_dis_ee = None
    reward_future_landing_dis = None
    reward_future_opponent_landing = None
    reward_future_landing_x_progress = None
    penalty_future_own_landing = None
    penalty_actual_own_table_after_hit = None
    reward_hit_ball_velocity_net = None
    reward_hit_net_clearance_progress = None
    reward_future_pass_net = None
    reward_table_success = None
    reward_actual_opponent_table_target = None
    reward_post_hit_net_progress = None
    reward_strike_window_touch_point = None
    reward_paddle_normal_alignment = None
    reward_paddle_swing_velocity = None
    penalty_unstable_hit = None
    penalty_forward_fall_during_strike = None
    penalty_hit_low_base_reset = None
    penalty_post_hit_low_base = None
    penalty_post_hit_trajectory_excess = None


@configclass
class A3Stage5ReadyEnvCfg(A3Stage4dEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_a3_stage5_ready_stance(self, A3_STAGE5_READY_BASE_POSE_RANGE)


@configclass
class A3Stage5ReadyEvalEnvCfg(A3Stage4dEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_a3_stage5_ready_stance(self, A3_STAGE5_READY_EVAL_BASE_POSE_RANGE)


@configclass
class A3Stage1BalanceMoveEnvCfg(A3Stage5ReadyEnvCfg):
    reward = A3Stage1BalanceMoveRewardCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 4.0
        self.scene.max_episode_length_s = 5.0
        self.commands.resampling_time_range = (5.0, 5.0)
        self.commands.rel_standing_envs = 1.0
        self.commands.rel_heading_envs = 1.0
        self.commands.heading_command = False
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.ranges.heading = (0.0, 0.0)
        _apply_a3_stage1_stance(self, A3_STAGE1_BASE_POSE_RANGE)
        _apply_a3_ball_ranges(self, A3_STAGE1_BALL_RANGES)
        self.robot.action_scale = A3_STAGE1_ACTION_SCALE_BY_JOINT.copy()
        self.normalization.clip_actions = 1.00
        self.robot.future_paddle_x_offset = A3_STAGE1_FUTURE_PADDLE_X_OFFSET
        self.robot.future_paddle_y_offset = A3_STAGE1_FUTURE_PADDLE_Y_OFFSET
        self.robot.future_invalid_robot_xy = A3_STAGE1_INVALID_ROBOT_XY
        self.robot.default_joint_pos_override = {
            **A3_STAGE1_UPPER_BODY_JOINT_POS,
            **A3_STAGE1_LOWER_BODY_JOINT_POS,
        }
        self.robot.use_fixed_target_xy_obs = True
        self.robot.fixed_target_xy_x_range = A3_STAGE1_TARGET_X_RANGE
        self.robot.fixed_target_xy_y_range = A3_STAGE1_TARGET_Y_RANGE
        self.robot.termination_min_base_z = A3_STAGE1_TERMINATION_MIN_BASE_Z
        self.robot.termination_max_flat_orientation_l2 = A3_STAGE1_TERMINATION_MAX_FLAT_ORIENTATION_L2
        self.robot.termination_robot_x_range = A3_STAGE1_TERMINATION_ROBOT_X_RANGE
        self.robot.termination_robot_y_range = A3_STAGE1_TERMINATION_ROBOT_Y_RANGE
        self.robot.stage1_bad_posture_min_base_z = A3_STAGE1_BAD_POSTURE_MIN_BASE_Z
        self.robot.stage1_bad_posture_max_flat_orientation_l2 = A3_STAGE1_BAD_POSTURE_MAX_FLAT_ORIENTATION_L2
        self.robot.stage1_bad_posture_max_steps = A3_STAGE1_BAD_POSTURE_MAX_STEPS
        self.robot.stage1_recovery_obs = True
        self.robot.actor_root_lin_vel_obs = True
        self.robot.stage1_recovery_target_z = 0.965
        self.robot.stage1_recovery_flat_deadband = 0.018
        self.ball.contact_threshold = 0.02
        self.ball.ball_max_eposide_length = 999999999.0
        self.ball.ball_reset_repeat = 1
        self.ball.max_serve_per_episode = 1_000_000
        self.domain_rand.events.reset_locomotion_joints.params["position_range"] = (
            A3_STAGE1_LOCOMOTION_JOINT_RESET_SCALE_RANGE
        )
        self.domain_rand.events.reset_manipulation_joints.params["position_range"] = (
            A3_STAGE1_MANIPULATION_JOINT_RESET_OFFSET_RANGE
        )
        self.domain_rand.events.push_robot = None
        if A3_STAGE1_ENABLE_INTERVAL_PUSH:
            self.domain_rand.events.push_robot = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(0.8, 1.2),
                params={
                    "velocity_range": A3_STAGE1_PUSH_VELOCITY_RANGE,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            )


@configclass
class A3Stage1BalanceMoveEvalEnvCfg(A3Stage5ReadyEvalEnvCfg):
    reward = A3Stage1BalanceMoveRewardCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 4.0
        self.scene.max_episode_length_s = 5.0
        _apply_a3_stage1_stance(self, A3_STAGE1_EVAL_BASE_POSE_RANGE)
        _apply_a3_ball_ranges(self, A3_STAGE1_BALL_RANGES)
        self.robot.action_scale = A3_STAGE1_ACTION_SCALE_BY_JOINT.copy()
        self.normalization.clip_actions = 1.00
        self.robot.future_paddle_x_offset = A3_STAGE1_FUTURE_PADDLE_X_OFFSET
        self.robot.future_paddle_y_offset = A3_STAGE1_FUTURE_PADDLE_Y_OFFSET
        self.robot.future_invalid_robot_xy = A3_STAGE1_INVALID_ROBOT_XY
        self.robot.default_joint_pos_override = {
            **A3_STAGE1_UPPER_BODY_JOINT_POS,
            **A3_STAGE1_LOWER_BODY_JOINT_POS,
        }
        self.robot.use_fixed_target_xy_obs = True
        self.robot.fixed_target_xy_x_range = A3_STAGE1_TARGET_X_RANGE
        self.robot.fixed_target_xy_y_range = A3_STAGE1_TARGET_Y_RANGE
        self.robot.termination_min_base_z = A3_STAGE1_TERMINATION_MIN_BASE_Z
        self.robot.termination_max_flat_orientation_l2 = A3_STAGE1_TERMINATION_MAX_FLAT_ORIENTATION_L2
        self.robot.termination_robot_x_range = A3_STAGE1_TERMINATION_ROBOT_X_RANGE
        self.robot.termination_robot_y_range = A3_STAGE1_TERMINATION_ROBOT_Y_RANGE
        self.robot.stage1_bad_posture_min_base_z = A3_STAGE1_BAD_POSTURE_MIN_BASE_Z
        self.robot.stage1_bad_posture_max_flat_orientation_l2 = A3_STAGE1_BAD_POSTURE_MAX_FLAT_ORIENTATION_L2
        self.robot.stage1_bad_posture_max_steps = A3_STAGE1_BAD_POSTURE_MAX_STEPS
        self.robot.stage1_recovery_obs = True
        self.robot.actor_root_lin_vel_obs = True
        self.robot.stage1_recovery_target_z = 0.965
        self.robot.stage1_recovery_flat_deadband = 0.018
        self.ball.contact_threshold = 0.02
        self.ball.ball_max_eposide_length = 999999999.0
        self.ball.ball_reset_repeat = 1
        self.ball.max_serve_per_episode = 1_000_000


@configclass
class A3Stage1StandRewardCfg(A3Stage1BalanceMoveRewardCfg):
    energy_ankle = RewTerm(
        func=mdp.energy,
        weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch_joint", ".*ankle_roll_joint"])},
    )
    penalty_ankle_roll_effort_saturation = RewTerm(
        func=mdp.penalty_a3_joint_effort_saturation,
        weight=-55.0,
        params={
            "threshold": 0.64,
            "std": 0.14,
            "max_penalty": 4.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_roll_joint"]),
        },
    )
    penalty_ankle_pitch_effort_saturation = RewTerm(
        func=mdp.penalty_a3_joint_effort_saturation,
        weight=-12.0,
        params={
            "threshold": 0.88,
            "std": 0.16,
            "max_penalty": 3.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch_joint"]),
        },
    )
    reward_alive = RewTerm(
        func=mdp.reward_alive_height_gated,
        weight=4.0,
        params={"min_z": 0.930, "transition": 0.050, "floor": 0.02},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1200.0)
    lin_vel_x_l2 = RewTerm(func=mdp.lin_vel_x_l2, weight=-2.5)
    lin_vel_y_l2 = RewTerm(func=mdp.lin_vel_y_l2, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-80.0)
    reward_base_height_target = RewTerm(
        func=mdp.reward_robot_base_height_target,
        weight=260.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "std": 0.055},
    )
    penalty_base_height_target = RewTerm(
        func=mdp.penalty_robot_base_height_target_l2,
        weight=-25.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "deadband": 0.025, "std": 0.070, "max_penalty": 5.0},
    )
    reward_base_height_hold = RewTerm(
        func=mdp.reward_robot_base_height_target,
        weight=0.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "std": 0.080},
    )
    reward_base_height_recovery = RewTerm(
        func=mdp.reward_robot_base_height_target,
        weight=0.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "std": 0.100},
    )
    reward_upright_recovery = RewTerm(
        func=mdp.reward_upright_exp,
        weight=0.0,
        params={"std": 0.20},
    )
    reward_base_height_recovery_rate = RewTerm(
        func=mdp.reward_base_height_recovery_rate,
        weight=0.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "deadband": 0.018, "std": 0.070, "max_rate": 1.0},
    )
    reward_stage1_low_height_lift_rate = RewTerm(
        func=mdp.reward_stage1_low_height_lift_rate,
        weight=0.0,
        params={
            "min_z": 0.880,
            "healthy_z": 0.950,
            "min_up_rate": 0.015,
            "max_up_rate": 0.35,
            "max_reward": 1.0,
        },
    )
    penalty_stage1_low_height_drop_rate = RewTerm(
        func=mdp.penalty_stage1_low_height_drop_rate,
        weight=0.0,
        params={
            "min_z": 0.880,
            "healthy_z": 0.950,
            "max_safe_down_rate": 0.010,
            "max_down_rate": 0.35,
            "max_penalty": 1.0,
        },
    )
    reward_upright_recovery_rate = RewTerm(
        func=mdp.reward_upright_recovery_rate,
        weight=0.0,
        params={"deadband": 0.018, "std": 0.055, "max_rate": 1.0},
    )
    reward_stage1_soft_recovery = RewTerm(
        func=mdp.reward_stage1_soft_recovery,
        weight=0.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.965,
            "max_flat_l2": 0.095,
            "max_forward_gravity_x": 0.115,
            "height_std": 0.060,
            "flat_std": 0.060,
            "forward_std": 0.060,
            "max_rate": 1.0,
        },
    )
    reward_stage1_recovery_posture_score = RewTerm(
        func=mdp.reward_stage1_recovery_posture_score,
        weight=0.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.980,
            "max_flat_l2": 0.090,
            "max_forward_gravity_x": 0.105,
            "height_std": 0.060,
            "flat_std": 0.060,
            "forward_std": 0.060,
            "min_root_z": 0.940,
        },
    )
    reward_stage1_result_bound_recovery = RewTerm(
        func=mdp.reward_stage1_result_bound_recovery,
        weight=45.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.945,
            "healthy_min_z": 0.950,
            "max_flat_l2": 0.120,
            "healthy_flat_l2": 0.085,
            "max_abs_gravity_x": 0.140,
            "healthy_abs_gravity_x": 0.095,
            "height_std": 0.045,
            "flat_std": 0.045,
            "pitch_std": 0.050,
            "min_root_z": 0.875,
            "posture_progress_min_z": 0.885,
            "min_height_progress": 0.0010,
            "healthy_bonus": 0.50,
            "max_reward": 2.0,
        },
    )
    reward_stage1_hip_knee_recovery_action = RewTerm(
        func=mdp.reward_stage1_hip_knee_recovery_action,
        weight=0.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.965,
            "max_forward_gravity_x": 0.095,
            "height_scale": 0.110,
            "desired_scale": 0.55,
            "opposite_scale": 1.20,
            "opposite_weight": 0.65,
        },
    )
    reward_xy_anchor = RewTerm(
        func=mdp.reward_robot_xy_target_height_gated,
        weight=0.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "std": 0.120,
            "min_z": 0.940,
            "transition": 0.010,
            "floor": 0.0,
        },
    )
    penalty_stage1_xy_drift = RewTerm(
        func=mdp.penalty_robot_xy_drift,
        weight=0.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.035,
            "y_margin": 0.035,
            "std": 0.120,
            "max_penalty": 5.0,
        },
    )
    penalty_stage1_forward_x_velocity = RewTerm(
        func=mdp.penalty_robot_forward_x_velocity,
        weight=0.0,
        params={"max_vx": 0.015, "std": 0.120, "max_penalty": 4.0},
    )
    penalty_stage1_forward_x_velocity_bound = RewTerm(
        func=mdp.penalty_robot_forward_x_velocity_after_bound,
        weight=0.0,
        params={"min_x": -1.405, "max_vx": 0.0, "x_std": 0.080, "vx_std": 0.120, "max_penalty": 4.0},
    )
    reward_stage1_xy_return_velocity = RewTerm(
        func=mdp.reward_robot_axis_velocity_towards_target_height_gated,
        weight=0.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.035,
            "y_margin": 0.035,
            "max_x_speed": 0.080,
            "max_y_speed": 0.060,
            "x_weight": 0.70,
            "y_weight": 0.30,
            "min_z": 0.940,
            "transition": 0.010,
            "floor": 0.0,
        },
    )
    penalty_stage1_xy_away_velocity = RewTerm(
        func=mdp.penalty_robot_axis_velocity_away_from_target,
        weight=0.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.035,
            "y_margin": 0.035,
            "x_std": 0.130,
            "y_std": 0.130,
            "x_weight": 0.70,
            "y_weight": 0.30,
            "max_penalty": 5.0,
        },
    )
    penalty_height_margin = RewTerm(
        func=mdp.penalty_robot_low_base_height,
        weight=0.0,
        params={"min_z": 0.965, "std": 0.065, "max_penalty": 5.0},
    )
    penalty_low_base_height_barrier = RewTerm(
        func=mdp.penalty_robot_low_base_height_barrier,
        weight=-80.0,
        params={"soft_min_z": 0.900, "hard_min_z": 0.875, "power": 2.0, "max_penalty": 5.0},
    )
    penalty_low_base_height = RewTerm(
        func=mdp.penalty_robot_low_base_height,
        weight=-150.0,
        params={"min_z": 0.940, "std": 0.050, "max_penalty": 5.0},
    )
    penalty_flat_orientation_margin = RewTerm(
        func=mdp.penalty_flat_orientation_margin,
        weight=-95.0,
        params={"max_flat_l2": 0.105, "std": 0.070, "max_penalty": 5.0},
    )
    penalty_stage1_forward_pitch_margin = RewTerm(
        func=mdp.penalty_stage1_forward_pitch_margin,
        weight=0.0,
        params={"max_forward_gravity_x": 0.115, "std": 0.065, "max_penalty": 6.0},
    )
    penalty_stage1_bad_posture = RewTerm(
        func=mdp.penalty_stage1_bad_posture,
        weight=0.0,
        params={
            "min_base_z": 0.940,
            "max_flat_l2": 0.120,
            "height_std": 0.065,
            "flat_std": 0.070,
            "max_penalty": 7.0,
        },
    )
    penalty_stage1_unproductive_recovery_action = RewTerm(
        func=mdp.penalty_stage1_unproductive_recovery_action,
        weight=0.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.965,
            "max_flat_l2": 0.095,
            "max_abs_gravity_x": 0.115,
            "min_height_progress": 0.0015,
            "min_flat_progress": 0.0015,
            "min_pitch_progress": 0.0015,
            "action_l2_ref": 4.0,
            "max_penalty": 3.0,
        },
    )
    penalty_stage1_passive_low_posture = RewTerm(
        func=mdp.penalty_stage1_passive_low_posture,
        weight=0.0,
        params={
            "recover_below_z": 0.950,
            "min_root_z": 0.880,
            "min_height_progress": 0.0015,
            "min_hip_knee_action_l1": 0.24,
            "max_penalty": 3.0,
        },
    )
    reward_stage1_low_posture_hip_knee_activity = RewTerm(
        func=mdp.reward_stage1_low_posture_hip_knee_activity,
        weight=0.0,
        params={
            "recover_below_z": 0.950,
            "min_root_z": 0.880,
            "target_action_l1": 0.35,
            "max_reward": 1.0,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.035)
    action_l2 = RewTerm(func=mdp.action_l2, weight=0.0)
    action_l2_healthy = RewTerm(
        func=mdp.penalty_action_l2_healthy_gated,
        weight=-6.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "z_deadband": 0.030,
            "z_std": 0.075,
            "max_flat_l2": 0.095,
            "flat_std": 0.065,
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.040,
            "y_margin": 0.040,
            "xy_std": 0.120,
            "min_gate": 0.0,
            "target_delta_ref": 0.040,
            "max_penalty": 8.0,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=A3_UNDESIRED_CONTACT_BODY_NAMES,
            ),
            "threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES),
        },
    )
    reward_stage1_healthy_p0_hold = RewTerm(
        func=mdp.reward_stage1_healthy_p0_hold,
        weight=0.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "min_z": 0.950,
            "z_deadband": 0.030,
            "z_std": 0.050,
            "z_transition": 0.010,
            "max_flat_l2": 0.080,
            "flat_std": 0.035,
            "max_forward_gravity_x": 0.115,
            "forward_std": 0.055,
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.040,
            "y_margin": 0.040,
            "xy_std": 0.120,
        },
    )
    reward_standing_stability = RewTerm(
        func=mdp.reward_standing_stability_height_gated,
        weight=0.0,
        params={"min_z": 0.950, "transition": 0.010, "floor": 0.0, **_a3_stage1_stand_stability_params()},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_yaw_joint", ".*hip_roll_joint"])},
    )
    joint_deviation_leg_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*hip_pitch_joint", ".*knee_joint", ".*ankle_pitch_joint"],
            )
        },
    )
    joint_deviation_stage1_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*hip_.*_joint", ".*knee_joint", ".*ankle_.*_joint"],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
    )
    penalty_stage1_forward_x = None
    fly = RewTerm(
        func=mdp.fly,
        weight=-2.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES), "threshold": 1.0},
    )
    feet_orientation_L = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_Link")},
    )
    feet_orientation_R = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_Link")},
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES),
            "threshold": 500,
            "max_reward": 400,
        },
    )


@configclass
class A3Stage1RecoveryStandRewardCfg(A3Stage1StandRewardCfg):
    reward_alive = RewTerm(
        func=mdp.reward_alive_height_gated,
        weight=3.0,
        params={"min_z": 0.920, "transition": 0.040, "floor": 0.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1200.0)
    reward_base_height_target = RewTerm(
        func=mdp.reward_robot_base_height_target,
        weight=320.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "std": 0.050},
    )
    penalty_base_height_target = RewTerm(
        func=mdp.penalty_robot_base_height_target_l2,
        weight=-28.0,
        params={"target_z": A3_STAGE1_STAND_TARGET_Z, "deadband": 0.020, "std": 0.060, "max_penalty": 6.0},
    )
    reward_stage1_result_bound_recovery = RewTerm(
        func=mdp.reward_stage1_result_bound_recovery,
        weight=70.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.955,
            "healthy_min_z": 0.950,
            "max_flat_l2": 0.110,
            "healthy_flat_l2": 0.080,
            "max_abs_gravity_x": 0.130,
            "healthy_abs_gravity_x": 0.090,
            "height_std": 0.050,
            "flat_std": 0.050,
            "pitch_std": 0.055,
            "min_root_z": 0.900,
            "posture_progress_min_z": 0.920,
            "min_height_progress": 0.0012,
            "healthy_bonus": 0.60,
            "max_reward": 2.2,
        },
    )
    reward_stage1_low_height_lift_rate = RewTerm(
        func=mdp.reward_stage1_low_height_lift_rate,
        weight=35.0,
        params={
            "min_z": 0.890,
            "healthy_z": 0.950,
            "min_up_rate": 0.012,
            "max_up_rate": 0.30,
            "max_reward": 1.0,
        },
    )
    penalty_stage1_low_height_drop_rate = RewTerm(
        func=mdp.penalty_stage1_low_height_drop_rate,
        weight=-35.0,
        params={
            "min_z": 0.890,
            "healthy_z": 0.950,
            "max_safe_down_rate": 0.008,
            "max_down_rate": 0.32,
            "max_penalty": 1.0,
        },
    )
    penalty_low_base_height_barrier = RewTerm(
        func=mdp.penalty_robot_low_base_height_barrier,
        weight=-180.0,
        params={"soft_min_z": 0.920, "hard_min_z": 0.875, "power": 2.0, "max_penalty": 6.0},
    )
    penalty_low_base_height = RewTerm(
        func=mdp.penalty_robot_low_base_height,
        weight=-240.0,
        params={"min_z": 0.945, "std": 0.040, "max_penalty": 6.0},
    )
    penalty_flat_orientation_margin = RewTerm(
        func=mdp.penalty_flat_orientation_margin,
        weight=-115.0,
        params={"max_flat_l2": 0.100, "std": 0.070, "max_penalty": 5.0},
    )
    penalty_stage1_forward_pitch_margin = RewTerm(
        func=mdp.penalty_stage1_forward_pitch_margin,
        weight=-18.0,
        params={"max_forward_gravity_x": 0.115, "std": 0.070, "max_penalty": 5.0},
    )
    penalty_stage1_passive_low_posture = RewTerm(
        func=mdp.penalty_stage1_passive_low_posture,
        weight=-25.0,
        params={
            "recover_below_z": 0.940,
            "min_root_z": 0.890,
            "min_height_progress": 0.0010,
            "min_hip_knee_action_l1": 0.16,
            "max_penalty": 2.5,
        },
    )
    penalty_stage1_unproductive_recovery_action = RewTerm(
        func=mdp.penalty_stage1_unproductive_recovery_action,
        weight=-6.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "recover_below_z": 0.945,
            "max_flat_l2": 0.105,
            "max_abs_gravity_x": 0.120,
            "min_height_progress": 0.0012,
            "min_flat_progress": 0.0012,
            "min_pitch_progress": 0.0012,
            "action_l2_ref": 3.0,
            "max_penalty": 3.0,
        },
    )
    reward_stage1_healthy_p0_hold = RewTerm(
        func=mdp.reward_stage1_healthy_p0_hold,
        weight=80.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "min_z": 0.948,
            "z_deadband": 0.030,
            "z_std": 0.055,
            "z_transition": 0.012,
            "max_flat_l2": 0.080,
            "flat_std": 0.040,
            "max_forward_gravity_x": 0.105,
            "forward_std": 0.055,
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.045,
            "y_margin": 0.045,
            "xy_std": 0.130,
        },
    )
    reward_standing_stability = RewTerm(
        func=mdp.reward_standing_stability_height_gated,
        weight=45.0,
        params={"min_z": 0.948, "transition": 0.015, "floor": 0.0, **_a3_stage1_stand_stability_params()},
    )
    action_l2_healthy = RewTerm(
        func=mdp.penalty_action_l2_healthy_gated,
        weight=-8.0,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "z_deadband": 0.030,
            "z_std": 0.075,
            "max_flat_l2": 0.090,
            "flat_std": 0.065,
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.045,
            "y_margin": 0.045,
            "xy_std": 0.130,
            "min_gate": 0.0,
            "target_delta_ref": 0.035,
            "max_penalty": 8.0,
        },
    )
    penalty_target_delta_l2 = RewTerm(
        func=mdp.penalty_target_delta_l2,
        weight=-18.0,
        params={"target_delta_ref": 0.040, "max_penalty": 12.0},
    )
    penalty_knee_flexion_target_bias = RewTerm(
        func=mdp.penalty_knee_flexion_target_bias,
        weight=-90.0,
        params={"deadband": 0.004, "target_delta_ref": 0.030, "max_penalty": 4.0},
    )


@configclass
class A3Stage1LateralMoveRewardCfg(A3Stage1StandRewardCfg):
    reward_xy_anchor = RewTerm(
        func=mdp.reward_robot_xy_target_stability_gated,
        weight=12.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "std": 0.070,
            "gate_floor": 0.0,
            "score_kwargs": _a3_stage1_stand_score_kwargs(),
        },
    )
    penalty_stage1_xy_drift = RewTerm(
        func=mdp.penalty_robot_xy_drift,
        weight=-3.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.025,
            "y_margin": 0.030,
            "std": 0.100,
            "max_penalty": 3.0,
        },
    )
    reward_stage1_xy_return_velocity = RewTerm(
        func=mdp.reward_robot_axis_velocity_towards_target_stability_gated,
        weight=0.0,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.025,
            "y_margin": 0.018,
            "max_x_speed": 0.040,
            "max_y_speed": 0.060,
            "x_weight": 0.20,
            "y_weight": 0.80,
            "gate_floor": 0.0,
            "score_kwargs": _a3_stage1_stand_score_kwargs(),
        },
    )
    penalty_stage1_xy_away_velocity = RewTerm(
        func=mdp.penalty_robot_axis_velocity_away_from_target,
        weight=-0.5,
        params={
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.025,
            "y_margin": 0.018,
            "x_std": 0.140,
            "y_std": 0.140,
            "x_weight": 0.20,
            "y_weight": 0.80,
            "max_penalty": 5.0,
        },
    )
    action_l2_healthy = RewTerm(
        func=mdp.penalty_action_l2_healthy_gated,
        weight=-2.5,
        params={
            "target_z": A3_STAGE1_STAND_TARGET_Z,
            "z_deadband": 0.030,
            "z_std": 0.075,
            "max_flat_l2": 0.095,
            "flat_std": 0.065,
            "target_xy": A3_STAGE1_INVALID_ROBOT_XY,
            "x_margin": 0.035,
            "y_margin": 0.035,
            "xy_std": 0.120,
            "min_gate": 0.0,
            "target_delta_ref": 0.045,
            "max_penalty": 10.0,
        },
    )


@configclass
class A3Stage1StandCurriculumCfg(CurriculumCfg):
    action_scale_to_nominal = CurrTerm(
        func=mdp.modify_action_scale_linear,
        params={
            "start_scale": 0.25,
            "target_scale": 1.00,
            "start_step": 0,
            "end_step": 96_000,
        },
    )


@configclass
class A3Stage1StandEnvCfg(A3Stage1BalanceMoveEnvCfg):
    reward = A3Stage1StandRewardCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_a3_stage1_stand_stance(self, A3_STAGE1_STAND_BASE_POSE_RANGE)
        self.noise.add_noise = False
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.physics_material = None
        if self.domain_rand.events.add_base_mass is not None:
            self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.domain_rand.perception_delay.enable = False
        self.domain_rand.action_delay.enable = False
        self.robot.action_scale = A3_STAGE1_STAND_ACTION_SCALE_BY_JOINT.copy()
        self.normalization.clip_actions = 1.00
        self.robot.default_joint_pos_override = {
            **A3_STAGE1_STAND_INIT_UPPER_BODY_JOINT_POS,
            **A3_STAGE1_STAND_LOWER_BODY_JOINT_POS,
        }
        self.robot.termination_min_base_z = 0.875
        self.robot.termination_max_flat_orientation_l2 = 0.360
        self.robot.stage1_bad_posture_min_base_z = 0.900
        self.robot.stage1_bad_posture_max_flat_orientation_l2 = 0.205
        self.robot.stage1_bad_posture_max_steps = 0
        self.robot.stage1_recovery_target_z = A3_STAGE1_STAND_TARGET_Z
        self.robot.action_health_gate_enable = True
        self.robot.action_health_gate_target_z = 0.945
        self.robot.action_health_gate_height_band = 0.050
        self.robot.action_health_gate_flat_l2_deadband = 0.060
        self.robot.action_health_gate_flat_l2_band = 0.100
        self.robot.action_health_gate_min_scale = 0.0


@configclass
class A3Stage1StandEvalEnvCfg(A3Stage1BalanceMoveEvalEnvCfg):
    reward = A3Stage1StandRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_a3_stage1_stand_stance(self, A3_STAGE1_STAND_BASE_POSE_RANGE)
        self.noise.add_noise = False
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.physics_material = None
        if self.domain_rand.events.add_base_mass is not None:
            self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.domain_rand.perception_delay.enable = False
        self.domain_rand.action_delay.enable = False
        self.robot.action_scale = A3_STAGE1_STAND_ACTION_SCALE_BY_JOINT.copy()
        self.normalization.clip_actions = 1.00
        self.robot.default_joint_pos_override = {
            **A3_STAGE1_STAND_INIT_UPPER_BODY_JOINT_POS,
            **A3_STAGE1_STAND_LOWER_BODY_JOINT_POS,
        }
        self.robot.termination_min_base_z = 0.875
        self.robot.termination_max_flat_orientation_l2 = 0.360
        self.robot.stage1_bad_posture_min_base_z = 0.900
        self.robot.stage1_bad_posture_max_flat_orientation_l2 = 0.205
        self.robot.stage1_bad_posture_max_steps = 0
        self.robot.stage1_recovery_target_z = A3_STAGE1_STAND_TARGET_Z
        self.robot.action_health_gate_enable = True
        self.robot.action_health_gate_target_z = 0.945
        self.robot.action_health_gate_height_band = 0.050
        self.robot.action_health_gate_flat_l2_deadband = 0.060
        self.robot.action_health_gate_flat_l2_band = 0.100
        self.robot.action_health_gate_min_scale = 0.0


@configclass
class A3Stage1RecoveryStandEnvCfg(A3Stage1StandEnvCfg):
    reward = A3Stage1RecoveryStandRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (0.16, 0.16),
            "y": (0.35, 0.35),
            "roll": (-0.014, 0.014),
            "pitch": (-0.020, 0.012),
            "yaw": (-0.003, 0.003),
        }
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (-0.012, 0.012),
            "y": (-0.010, 0.010),
            "z": (-0.015, 0.012),
            "roll": (-0.040, 0.040),
            "pitch": (-0.050, 0.050),
            "yaw": (-0.012, 0.012),
        }
        self.robot.termination_min_base_z = 0.875
        self.robot.termination_max_flat_orientation_l2 = 0.360
        self.robot.stage1_bad_posture_max_steps = 0
        self.robot.action_health_gate_target_z = 0.950
        self.robot.action_health_gate_height_band = 0.060
        self.robot.action_health_gate_flat_l2_deadband = 0.050
        self.robot.action_health_gate_flat_l2_band = 0.120
        self.robot.action_health_gate_min_scale = 0.04


@configclass
class A3Stage1RecoveryStandEvalEnvCfg(A3Stage1StandEvalEnvCfg):
    reward = A3Stage1RecoveryStandRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.termination_min_base_z = 0.875
        self.robot.termination_max_flat_orientation_l2 = 0.360
        self.robot.stage1_bad_posture_max_steps = 0
        self.robot.action_health_gate_target_z = 0.950
        self.robot.action_health_gate_height_band = 0.060
        self.robot.action_health_gate_flat_l2_deadband = 0.050
        self.robot.action_health_gate_flat_l2_band = 0.120
        self.robot.action_health_gate_min_scale = 0.04


@configclass
class A3Stage1LateralMoveEnvCfg(A3Stage1StandEnvCfg):
    reward = A3Stage1LateralMoveRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.fixed_target_xy_x_range = A3_STAGE1_LATERAL_TARGET_X_RANGE
        self.robot.fixed_target_xy_y_range = A3_STAGE1_LATERAL_TARGET_Y_RANGE
        self.robot.action_health_gate_target_z = 0.950
        self.robot.action_health_gate_height_band = 0.050
        self.robot.action_health_gate_min_scale = 0.0


@configclass
class A3Stage1LateralMoveEvalEnvCfg(A3Stage1StandEvalEnvCfg):
    reward = A3Stage1LateralMoveRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.fixed_target_xy_x_range = A3_STAGE1_LATERAL_TARGET_X_RANGE
        self.robot.fixed_target_xy_y_range = A3_STAGE1_LATERAL_TARGET_Y_RANGE
        self.robot.action_health_gate_target_z = 0.950
        self.robot.action_health_gate_height_band = 0.050
        self.robot.action_health_gate_min_scale = 0.0


@configclass
class A3Stage5bRewardCfg(A3Stage4dRewardCfg):
    reward_contact = RewTerm(func=mdp.reward_contact, weight=150.0)
    penalty_future_own_landing = RewTerm(
        func=mdp.penalty_future_own_landing_after_hit,
        weight=-15.0,
        params={"max_x": 0.0},
    )
    penalty_actual_own_table_after_hit = RewTerm(
        func=mdp.penalty_own_table_after_paddle_hit,
        weight=-20.0,
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=260.0,
        params={
            "vx_target": 2.4,
            "vz_target": 1.2,
            "z_target": 1.02,
            "z_std": 0.45,
            "min_vx": 0.05,
            "max_t_net": 1.6,
            "t_std": 0.9,
            "vx_weight": 0.70,
            "vz_weight": 0.10,
            "z_weight": 0.15,
            "t_weight": 0.05,
        },
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress,
        weight=140.0,
        params={
            "min_vx": 0.05,
            "vx_target": 2.2,
            "min_z": 0.76,
            "target_z": 1.00,
            "z_std": 0.55,
            "max_t_net": 1.8,
            "t_std": 1.0,
            "vx_weight": 0.75,
            "time_weight": 0.25,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.55, "z_target": 0.76 + 0.25},
        weight=260.0,
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress,
        weight=90.0,
        params={
            "min_vx": 0.05,
            "vx_target": 2.8,
            "vz_target": 1.1,
            "x_start": -1.45,
            "max_reward_x": -0.95,
            "net_x": 0.0,
            "net_z_target": 1.00,
            "min_clear_z": 0.76,
            "z_std": 0.55,
            "max_t_net": 1.6,
            "landing_min_x": -1.5,
            "landing_target_x": 1.15,
            "y_target": 0.0,
            "y_std": 0.85,
            "vy_std": 2.2,
            "vx_weight": 0.35,
            "vz_weight": 0.10,
            "x_weight": 0.10,
            "z_weight": 0.25,
            "landing_weight": 0.10,
            "y_weight": 0.10,
        },
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=220.0)
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target,
        weight=100.0,
        params={"target_x": 1.15, "target_y": 0.0, "x_std": 0.8, "y_std": 0.6},
    )
    penalty_hit_low_base_reset = RewTerm(
        func=mdp.penalty_hit_low_base_reset,
        weight=-25.0,
        params={"min_base_z": 0.50},
    )


@configclass
class A3Stage5bEnvCfg(A3Stage5ReadyEnvCfg):
    reward = A3Stage5bRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.action_scale = A3_STAGE5B_ACTION_SCALE_BY_JOINT.copy()


@configclass
class A3Stage5bEvalEnvCfg(A3Stage5ReadyEvalEnvCfg):
    reward = A3Stage5bRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.action_scale = A3_STAGE5B_ACTION_SCALE_BY_JOINT.copy()


@configclass
class A3Stage5cRewardCfg(A3Stage5bRewardCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1600.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-140.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=A3_UNDESIRED_CONTACT_BODY_NAMES,
            ),
            "threshold": 1.0,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    hit_unstable_support = RewTerm(
        func=mdp.hit_unstable_support,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES)},
    )
    feet_orientation_L = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_ankle_roll_Link")},
    )
    feet_orientation_R = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_ankle_roll_Link")},
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES), "threshold": 0.2},
    )
    feet_really_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=A3_FEET_BODY_NAMES), "threshold": 0.15},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=A3_FEET_BODY_NAMES)},
    )
    reward_future_touch_point = RewTerm(
        func=mdp.reward_future_touch_point_target,
        weight=10.0,
        params={"std_ee": 0.5, "threshold": 0.03},
    )
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=8.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_contact = RewTerm(func=mdp.reward_contact, weight=90.0)
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=75.0,
        params={
            "vx_target": 2.6,
            "vz_target": 1.35,
            "z_target": 1.08,
            "z_std": 0.42,
            "min_vx": 0.05,
            "max_t_net": 1.6,
            "t_std": 0.9,
            "vx_weight": 0.55,
            "vz_weight": 0.20,
            "z_weight": 0.20,
            "t_weight": 0.05,
        },
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress,
        weight=50.0,
        params={
            "min_vx": 0.05,
            "vx_target": 2.4,
            "min_z": 0.78,
            "target_z": 1.07,
            "z_std": 0.48,
            "max_t_net": 1.8,
            "t_std": 1.0,
            "vx_weight": 0.60,
            "time_weight": 0.40,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.45, "z_target": 0.76 + 0.30},
        weight=60.0,
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress,
        weight=35.0,
        params={
            "min_vx": 0.05,
            "vx_target": 3.0,
            "vz_target": 1.3,
            "x_start": -1.45,
            "max_reward_x": -0.85,
            "net_x": 0.0,
            "net_z_target": 1.08,
            "min_clear_z": 0.78,
            "z_std": 0.48,
            "max_t_net": 1.6,
            "landing_min_x": -1.5,
            "landing_target_x": 1.15,
            "y_target": 0.0,
            "y_std": 0.85,
            "vy_std": 2.2,
            "vx_weight": 0.30,
            "vz_weight": 0.15,
            "x_weight": 0.05,
            "z_weight": 0.30,
            "landing_weight": 0.10,
            "y_weight": 0.10,
        },
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=35.0)
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target,
        weight=25.0,
        params={"target_x": 1.15, "target_y": 0.0, "x_std": 0.8, "y_std": 0.6},
    )
    penalty_future_own_landing = RewTerm(
        func=mdp.penalty_future_own_landing_after_hit,
        weight=-5.0,
        params={"max_x": 0.0},
    )
    penalty_actual_own_table_after_hit = RewTerm(
        func=mdp.penalty_own_table_after_paddle_hit,
        weight=-5.0,
    )
    penalty_hit_low_base_reset = RewTerm(
        func=mdp.penalty_hit_low_base_reset,
        weight=-120.0,
        params={"min_base_z": 0.50},
    )


@configclass
class A3Stage5cCurriculumCfg(CurriculumCfg):
    termination_penalty_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "termination_penalty", "target_weight": -1000.0, "start_step": 4000, "end_step": 24000},
    )
    undesired_contacts_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "undesired_contacts", "target_weight": -90.0, "start_step": 4000, "end_step": 24000},
    )
    flat_orientation_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "flat_orientation_l2", "target_weight": -1.5, "start_step": 4000, "end_step": 24000},
    )
    hit_unstable_support_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "hit_unstable_support", "target_weight": -10.0, "start_step": 4000, "end_step": 24000},
    )
    feet_orientation_l_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "feet_orientation_L", "target_weight": -4.0, "start_step": 4000, "end_step": 24000},
    )
    feet_orientation_r_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "feet_orientation_R", "target_weight": -4.0, "start_step": 4000, "end_step": 24000},
    )
    feet_too_near_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "feet_too_near", "target_weight": -1.5, "start_step": 4000, "end_step": 24000},
    )
    feet_really_too_near_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "feet_really_too_near", "target_weight": -10.0, "start_step": 4000, "end_step": 24000},
    )
    feet_stumble_to_nominal = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "feet_stumble", "target_weight": -2.0, "start_step": 4000, "end_step": 24000},
    )
    low_base_reset_to_mid = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "penalty_hit_low_base_reset", "target_weight": -45.0, "start_step": 4000, "end_step": 24000},
    )
    future_touch_point_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_future_touch_point", "target_weight": 14.0, "start_step": 4000, "end_step": 24000},
    )
    future_dis_ee_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_future_dis_ee", "target_weight": 10.0, "start_step": 4000, "end_step": 24000},
    )
    contact_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_contact", "target_weight": 150.0, "start_step": 4000, "end_step": 24000},
    )
    hit_velocity_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_hit_ball_velocity_net", "target_weight": 310.0, "start_step": 12000, "end_step": 60000},
    )
    hit_net_clearance_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "reward_hit_net_clearance_progress",
            "target_weight": 220.0,
            "start_step": 12000,
            "end_step": 60000,
        },
    )
    post_hit_net_progress_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_post_hit_net_progress", "target_weight": 170.0, "start_step": 12000, "end_step": 60000},
    )
    future_pass_net_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_future_pass_net", "target_weight": 320.0, "start_step": 12000, "end_step": 60000},
    )
    table_success_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_table_success", "target_weight": 260.0, "start_step": 12000, "end_step": 60000},
    )
    opponent_table_target_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "reward_actual_opponent_table_target",
            "target_weight": 140.0,
            "start_step": 12000,
            "end_step": 60000,
        },
    )
    future_own_landing_penalty_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "penalty_future_own_landing", "target_weight": -25.0, "start_step": 24000, "end_step": 60000},
    )
    actual_own_table_penalty_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={
            "term_name": "penalty_actual_own_table_after_hit",
            "target_weight": -30.0,
            "start_step": 24000,
            "end_step": 60000,
        },
    )


@configclass
class A3Stage5cEnvCfg(A3Stage5ReadyEnvCfg):
    reward = A3Stage5cRewardCfg()
    curriculum = A3Stage5cCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.action_scale = A3_STAGE5B_ACTION_SCALE_BY_JOINT.copy()


@configclass
class A3Stage5cEvalEnvCfg(A3Stage5ReadyEvalEnvCfg):
    reward = A3Stage5cRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.action_scale = A3_STAGE5B_ACTION_SCALE_BY_JOINT.copy()


@configclass
class A3Stage5dRewardCfg(A3Stage5cRewardCfg):
    reward_future_touch_point = RewTerm(
        func=mdp.reward_future_touch_point_target,
        weight=8.0,
        params={"std_ee": 0.5, "threshold": 0.03},
    )
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=7.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_contact = RewTerm(func=mdp.reward_contact, weight=55.0)


@configclass
class A3Stage5dCurriculumCfg(A3Stage5cCurriculumCfg):
    ball_range_curriculum = CurrTerm(
        func=mdp.modify_ball_ranges_piecewise_linear,
        params={"phases": A3_STAGE5D_BALL_CURRICULUM_PHASES, "start_step": 0},
    )
    future_touch_point_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_future_touch_point", "target_weight": 10.0, "start_step": 4000, "end_step": 24000},
    )
    future_dis_ee_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_future_dis_ee", "target_weight": 8.0, "start_step": 4000, "end_step": 24000},
    )
    contact_up = CurrTerm(
        func=mdp.modify_reward_weight_linear,
        params={"term_name": "reward_contact", "target_weight": 90.0, "start_step": 4000, "end_step": 24000},
    )


@configclass
class A3Stage5dEnvCfg(A3Stage5cEnvCfg):
    reward = A3Stage5dRewardCfg()
    curriculum = A3Stage5dCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_a3_ball_ranges(self, A3_STAGE5D_BALL_START_RANGES)
        self.ball.contact_threshold = A3_STAGE5D_CONTACT_THRESHOLD


@configclass
class A3Stage5dEvalEnvCfg(A3Stage5cEvalEnvCfg):
    reward = A3Stage5dRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        final_ball_ranges = {
            key: A3_STAGE5D_BALL_CURRICULUM_PHASES[-1][key]
            for key in A3_STAGE5D_BALL_START_RANGES
        }
        _apply_a3_ball_ranges(self, final_ball_ranges)
        self.ball.contact_threshold = A3_STAGE5D_CONTACT_THRESHOLD


@configclass
class A3Stage5eRewardCfg(A3Stage5dRewardCfg):
    reward_standing_stability = RewTerm(
        func=mdp.reward_standing_stability,
        weight=12.0,
        params=_a3_stage5e_stability_params(),
    )
    penalty_unstable_hit = RewTerm(
        func=mdp.penalty_unstable_hit,
        weight=-60.0,
        params=_a3_stage5e_stability_params(),
    )
    reward_future_touch_point = RewTerm(
        func=mdp.reward_future_touch_point_target_stability_gated,
        weight=8.0,
        params=_a3_stage5e_gated_params(0.30, std_ee=0.5, threshold=0.03),
    )
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target_stability_gated,
        weight=7.0,
        params=_a3_stage5e_gated_params(0.30, std_ee=0.5, threshold=0.15),
    )
    reward_contact = RewTerm(
        func=mdp.reward_contact_stability_gated,
        weight=55.0,
        params=_a3_stage5e_stability_params(0.30),
    )
    reward_future_landing_x_progress = RewTerm(
        func=mdp.reward_future_landing_x_progress_stability_gated,
        weight=110.0,
        params=_a3_stage5e_gated_params(
            0.15,
            min_x=-3.0,
            target_x=1.15,
            target_y=0.0,
            y_std=0.9,
            y_weight=0.30,
        ),
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target_stability_gated,
        weight=75.0,
        params=_a3_stage5e_gated_params(
            0.15,
            vx_target=2.6,
            vz_target=1.35,
            z_target=1.08,
            z_std=0.42,
            min_vx=0.05,
            max_t_net=1.6,
            t_std=0.9,
            vx_weight=0.55,
            vz_weight=0.20,
            z_weight=0.20,
            t_weight=0.05,
        ),
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress_stability_gated,
        weight=50.0,
        params=_a3_stage5e_gated_params(
            0.15,
            min_vx=0.05,
            vx_target=2.4,
            min_z=0.78,
            target_z=1.07,
            z_std=0.48,
            max_t_net=1.8,
            t_std=1.0,
            vx_weight=0.60,
            time_weight=0.40,
        ),
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net_stability_gated,
        weight=60.0,
        params=_a3_stage5e_gated_params(0.15, std_h=0.45, z_target=0.76 + 0.30),
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress_stability_gated,
        weight=35.0,
        params=_a3_stage5e_post_hit_params(
            0.15,
            min_vx=0.05,
            vx_target=3.0,
            vz_target=1.3,
            x_start=-1.45,
            max_reward_x=-0.85,
            net_x=0.0,
            net_z_target=1.08,
            min_clear_z=0.78,
            z_std=0.48,
            max_t_net=1.6,
            landing_min_x=-1.5,
            landing_target_x=1.15,
            y_target=0.0,
            y_std=0.85,
            vy_std=2.2,
            vx_weight=0.30,
            vz_weight=0.15,
            x_weight=0.05,
            z_weight=0.30,
            landing_weight=0.10,
            y_weight=0.10,
        ),
    )
    reward_table_success = RewTerm(
        func=mdp.reward_table_success_stability_gated,
        weight=35.0,
        params=_a3_stage5e_stability_params(0.10),
    )
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target_stability_gated,
        weight=25.0,
        params=_a3_stage5e_gated_params(
            0.10,
            target_x=1.15,
            target_y=0.0,
            x_std=0.8,
            y_std=0.6,
        ),
    )


@configclass
class A3Stage5eEnvCfg(A3Stage5dEnvCfg):
    reward = A3Stage5eRewardCfg()


@configclass
class A3Stage5eEvalEnvCfg(A3Stage5dEvalEnvCfg):
    reward = A3Stage5eRewardCfg()


@configclass
class A3Stage5fRewardCfg(A3Stage5dRewardCfg):
    reward_standing_stability = RewTerm(
        func=mdp.reward_standing_stability,
        weight=18.0,
        params=_a3_stage5f_stability_params(),
    )
    penalty_unstable_hit = RewTerm(
        func=mdp.penalty_unstable_hit,
        weight=-18.0,
        params=_a3_stage5f_stability_params(),
    )
    reward_future_touch_point = RewTerm(
        func=mdp.reward_future_touch_point_target,
        weight=9.0,
        params={"std_ee": 0.5, "threshold": 0.03},
    )
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target,
        weight=8.0,
        params={"std_ee": 0.5, "threshold": 0.15},
    )
    reward_contact = RewTerm(func=mdp.reward_contact, weight=80.0)
    reward_future_landing_x_progress = RewTerm(
        func=mdp.reward_future_landing_x_progress,
        weight=120.0,
        params={"min_x": -3.0, "target_x": 1.15, "target_y": 0.0, "y_std": 0.9, "y_weight": 0.30},
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target_stability_gated,
        weight=90.0,
        params=_a3_stage5f_gated_params(
            0.35,
            vx_target=2.6,
            vz_target=1.35,
            z_target=1.08,
            z_std=0.42,
            min_vx=0.05,
            max_t_net=1.6,
            t_std=0.9,
            vx_weight=0.55,
            vz_weight=0.20,
            z_weight=0.20,
            t_weight=0.05,
        ),
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress_stability_gated,
        weight=60.0,
        params=_a3_stage5f_gated_params(
            0.35,
            min_vx=0.05,
            vx_target=2.4,
            min_z=0.78,
            target_z=1.07,
            z_std=0.48,
            max_t_net=1.8,
            t_std=1.0,
            vx_weight=0.60,
            time_weight=0.40,
        ),
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net_stability_gated,
        weight=75.0,
        params=_a3_stage5f_gated_params(0.35, std_h=0.45, z_target=0.76 + 0.30),
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress_stability_gated,
        weight=45.0,
        params=_a3_stage5f_post_hit_params(
            0.35,
            min_vx=0.05,
            vx_target=3.0,
            vz_target=1.3,
            x_start=-1.45,
            max_reward_x=-0.85,
            net_x=0.0,
            net_z_target=1.08,
            min_clear_z=0.78,
            z_std=0.48,
            max_t_net=1.6,
            landing_min_x=-1.5,
            landing_target_x=1.15,
            y_target=0.0,
            y_std=0.85,
            vy_std=2.2,
            vx_weight=0.30,
            vz_weight=0.15,
            x_weight=0.05,
            z_weight=0.30,
            landing_weight=0.10,
            y_weight=0.10,
        ),
    )
    reward_table_success = RewTerm(
        func=mdp.reward_table_success_stability_gated,
        weight=45.0,
        params=_a3_stage5f_stability_params(0.25),
    )
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target_stability_gated,
        weight=35.0,
        params=_a3_stage5f_gated_params(
            0.25,
            target_x=1.15,
            target_y=0.0,
            x_std=0.8,
            y_std=0.6,
        ),
    )


@configclass
class A3Stage5fCurriculumCfg(A3Stage5cCurriculumCfg):
    ball_range_curriculum = CurrTerm(
        func=mdp.modify_ball_ranges_by_ability,
        params={"phases": A3_STAGE5F_BALL_ABILITY_PHASES, "min_window_steps": 1200, "min_window_serves": 1024},
    )


@configclass
class A3Stage5fEnvCfg(A3Stage5dEnvCfg):
    reward = A3Stage5fRewardCfg()
    curriculum = A3Stage5fCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_a3_ball_ranges(self, A3_STAGE5F_BALL_ABILITY_PHASES[0]["ranges"])
        self.ball.contact_threshold = A3_STAGE5D_CONTACT_THRESHOLD
        self.robot.action_scale = A3_STAGE5F_ACTION_SCALE_BY_JOINT.copy()


@configclass
class A3Stage5fEvalEnvCfg(A3Stage5dEvalEnvCfg):
    reward = A3Stage5fRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_a3_ball_ranges(self, A3_STAGE5F_BALL_ABILITY_PHASES[0]["ranges"])
        self.ball.contact_threshold = A3_STAGE5D_CONTACT_THRESHOLD
        self.robot.action_scale = A3_STAGE5F_ACTION_SCALE_BY_JOINT.copy()


@configclass
class A3Stage5gEnvCfg(A3Stage5fEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot.future_paddle_x_offset = A3_STAGE5G_FUTURE_PADDLE_X_OFFSET
        self.robot.future_paddle_y_offset = A3_STAGE5G_FUTURE_PADDLE_Y_OFFSET
        self.robot.future_invalid_robot_xy = A3_STAGE5G_INVALID_ROBOT_XY


@configclass
class A3Stage5gEvalEnvCfg(A3Stage5fEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot.future_paddle_x_offset = A3_STAGE5G_FUTURE_PADDLE_X_OFFSET
        self.robot.future_paddle_y_offset = A3_STAGE5G_FUTURE_PADDLE_Y_OFFSET
        self.robot.future_invalid_robot_xy = A3_STAGE5G_INVALID_ROBOT_XY


@configclass
class A3Stage5gFixedBallEnvCfg(A3Stage5gEnvCfg):
    curriculum = CurriculumCfg()


@configclass
class A3Stage5gFixedBallEvalEnvCfg(A3Stage5gEvalEnvCfg):
    curriculum = CurriculumCfg()


@configclass
class A3Stage5gWideEnvCfg(A3Stage5gFixedBallEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_a3_ball_ranges(self, A3_STAGE5G_WIDE_BALL_RANGES)
        self.ball.contact_threshold = A3_STAGE5G_WIDE_CONTACT_THRESHOLD


@configclass
class A3Stage5gWideEvalEnvCfg(A3Stage5gFixedBallEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_a3_ball_ranges(self, A3_STAGE5G_WIDE_BALL_RANGES)
        self.ball.contact_threshold = A3_STAGE5G_WIDE_CONTACT_THRESHOLD


@configclass
class A3Stage5hHitQualityRewardCfg(A3Stage5fRewardCfg):
    reward_strike_window_touch_point = RewTerm(
        func=mdp.reward_strike_window_touch_point_stability_gated,
        weight=20.0,
        params=_a3_stage5f_gated_params(
            0.30,
            center_t=0.24,
            std_t=0.18,
            min_t=0.04,
            max_t=0.70,
            std_ee=0.34,
            threshold=0.03,
        ),
    )
    reward_paddle_normal_alignment = RewTerm(
        func=mdp.reward_paddle_normal_alignment_stability_gated,
        weight=16.0,
        params=_a3_stage5f_gated_params(
            0.30,
            local_normal=A3_STAGE5H_PADDLE_NORMAL_AXIS,
            center_t=0.24,
            std_t=0.18,
            min_t=0.04,
            max_t=0.70,
            dist_std=0.85,
            align_power=1.5,
        ),
    )
    reward_paddle_swing_velocity = RewTerm(
        func=mdp.reward_paddle_swing_velocity_target_stability_gated,
        weight=22.0,
        params=_a3_stage5f_gated_params(
            0.30,
            target_x=1.15,
            target_y=0.0,
            target_z=1.05,
            target_speed=1.15,
            min_speed=0.04,
            local_normal=A3_STAGE5H_PADDLE_NORMAL_AXIS,
            normal_floor=0.30,
            center_t=0.22,
            std_t=0.16,
            min_t=0.04,
            max_t=0.60,
            dist_std=0.70,
        ),
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target_stability_gated,
        weight=120.0,
        params=_a3_stage5f_gated_params(
            0.35,
            vx_target=3.1,
            vz_target=1.15,
            z_target=1.05,
            z_std=0.34,
            min_vx=0.08,
            max_t_net=1.35,
            t_std=0.75,
            vx_weight=0.62,
            vz_weight=0.08,
            z_weight=0.25,
            t_weight=0.05,
        ),
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress_stability_gated,
        weight=85.0,
        params=_a3_stage5f_gated_params(
            0.35,
            min_vx=0.08,
            vx_target=2.7,
            min_z=0.78,
            target_z=1.04,
            z_std=0.38,
            max_t_net=1.55,
            t_std=0.85,
            vx_weight=0.68,
            time_weight=0.32,
        ),
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net_stability_gated,
        weight=100.0,
        params=_a3_stage5f_gated_params(0.35, std_h=0.36, z_target=0.76 + 0.28),
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress_stability_gated,
        weight=65.0,
        params=_a3_stage5f_post_hit_params(
            0.35,
            min_vx=0.08,
            vx_target=3.2,
            vz_target=1.05,
            x_start=-1.45,
            max_reward_x=-0.85,
            net_x=0.0,
            net_z_target=1.04,
            min_clear_z=0.78,
            z_std=0.38,
            max_t_net=1.35,
            landing_min_x=-1.5,
            landing_target_x=1.15,
            y_target=0.0,
            y_std=0.75,
            vy_std=1.8,
            vx_weight=0.38,
            vz_weight=0.05,
            x_weight=0.05,
            z_weight=0.30,
            landing_weight=0.12,
            y_weight=0.10,
        ),
    )


@configclass
class A3Stage5hHitQualityEnvCfg(A3Stage5gWideEnvCfg):
    reward = A3Stage5hHitQualityRewardCfg()


@configclass
class A3Stage5hHitQualityEvalEnvCfg(A3Stage5gWideEvalEnvCfg):
    reward = A3Stage5hHitQualityRewardCfg()


@configclass
class A3Stage5iStableHitQualityRewardCfg(A3Stage5hHitQualityRewardCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2000.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)
    reward_standing_stability = RewTerm(
        func=mdp.reward_standing_stability,
        weight=55.0,
        params=_a3_stage5i_stability_params(),
    )
    penalty_unstable_hit = RewTerm(
        func=mdp.penalty_unstable_hit,
        weight=-70.0,
        params=_a3_stage5i_stability_params(),
    )
    penalty_forward_fall_during_strike = RewTerm(
        func=mdp.penalty_a3_forward_fall_during_strike,
        weight=-85.0,
        params={
            "center_t": 0.24,
            "std_t": 0.18,
            "min_t": 0.02,
            "max_t": 0.75,
            "max_root_x": -1.48,
            "max_forward_vx": 0.35,
            "max_tilt": 0.45,
            "min_base_z": 0.78,
        },
    )
    reward_future_touch_point = RewTerm(
        func=mdp.reward_future_touch_point_target_stability_gated,
        weight=7.0,
        params=_a3_stage5i_gated_params(0.12, std_ee=0.5, threshold=0.03),
    )
    reward_future_dis_ee = RewTerm(
        func=mdp.reward_future_ee_target_stability_gated,
        weight=6.0,
        params=_a3_stage5i_gated_params(0.12, std_ee=0.5, threshold=0.15),
    )
    reward_contact = RewTerm(
        func=mdp.reward_contact_stability_gated,
        weight=75.0,
        params=_a3_stage5i_stability_params(0.05),
    )
    reward_future_landing_x_progress = RewTerm(
        func=mdp.reward_future_landing_x_progress_stability_gated,
        weight=90.0,
        params=_a3_stage5i_gated_params(
            0.08,
            min_x=-3.0,
            target_x=1.15,
            target_y=0.0,
            y_std=0.9,
            y_weight=0.30,
        ),
    )
    reward_strike_window_touch_point = RewTerm(
        func=mdp.reward_strike_window_touch_point_stability_gated,
        weight=16.0,
        params=_a3_stage5i_gated_params(
            0.05,
            center_t=0.24,
            std_t=0.18,
            min_t=0.04,
            max_t=0.70,
            std_ee=0.34,
            threshold=0.03,
        ),
    )
    reward_paddle_normal_alignment = RewTerm(
        func=mdp.reward_paddle_normal_alignment_stability_gated,
        weight=14.0,
        params=_a3_stage5i_gated_params(
            0.05,
            local_normal=A3_STAGE5H_PADDLE_NORMAL_AXIS,
            center_t=0.24,
            std_t=0.18,
            min_t=0.04,
            max_t=0.70,
            dist_std=0.85,
            align_power=1.5,
        ),
    )
    reward_paddle_swing_velocity = RewTerm(
        func=mdp.reward_paddle_swing_velocity_target_stability_gated,
        weight=14.0,
        params=_a3_stage5i_gated_params(
            0.05,
            target_x=1.15,
            target_y=0.0,
            target_z=1.05,
            target_speed=1.05,
            min_speed=0.04,
            local_normal=A3_STAGE5H_PADDLE_NORMAL_AXIS,
            normal_floor=0.30,
            center_t=0.22,
            std_t=0.16,
            min_t=0.04,
            max_t=0.60,
            dist_std=0.70,
        ),
    )
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target_stability_gated,
        weight=110.0,
        params=_a3_stage5i_gated_params(
            0.08,
            vx_target=3.1,
            vz_target=1.15,
            z_target=1.05,
            z_std=0.34,
            min_vx=0.08,
            max_t_net=1.35,
            t_std=0.75,
            vx_weight=0.62,
            vz_weight=0.08,
            z_weight=0.25,
            t_weight=0.05,
        ),
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress_stability_gated,
        weight=75.0,
        params=_a3_stage5i_gated_params(
            0.08,
            min_vx=0.08,
            vx_target=2.7,
            min_z=0.78,
            target_z=1.04,
            z_std=0.38,
            max_t_net=1.55,
            t_std=0.85,
            vx_weight=0.68,
            time_weight=0.32,
        ),
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net_stability_gated,
        weight=95.0,
        params=_a3_stage5i_gated_params(0.08, std_h=0.36, z_target=0.76 + 0.28),
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress_stability_gated,
        weight=60.0,
        params=_a3_stage5i_post_hit_params(
            0.08,
            min_vx=0.08,
            vx_target=3.2,
            vz_target=1.05,
            x_start=-1.45,
            max_reward_x=-0.85,
            net_x=0.0,
            net_z_target=1.04,
            min_clear_z=0.78,
            z_std=0.38,
            max_t_net=1.35,
            landing_min_x=-1.5,
            landing_target_x=1.15,
            y_target=0.0,
            y_std=0.75,
            vy_std=1.8,
            vx_weight=0.38,
            vz_weight=0.05,
            x_weight=0.05,
            z_weight=0.30,
            landing_weight=0.12,
            y_weight=0.10,
        ),
    )
    reward_table_success = RewTerm(
        func=mdp.reward_table_success_stability_gated,
        weight=55.0,
        params=_a3_stage5i_stability_params(0.05),
    )
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target_stability_gated,
        weight=45.0,
        params=_a3_stage5i_gated_params(
            0.05,
            target_x=1.15,
            target_y=0.0,
            x_std=0.8,
            y_std=0.6,
        ),
    )


@configclass
class A3Stage5iStableHitQualityEnvCfg(A3Stage5hHitQualityEnvCfg):
    reward = A3Stage5iStableHitQualityRewardCfg()


@configclass
class A3Stage5iStableHitQualityEvalEnvCfg(A3Stage5hHitQualityEvalEnvCfg):
    reward = A3Stage5iStableHitQualityRewardCfg()


@configclass
class A3Stage4eRewardCfg(A3Stage4dRewardCfg):
    reward_future_opponent_landing = RewTerm(
        func=mdp.reward_future_opponent_landing_target,
        weight=220.0,
        params={"target_x": 1.15, "target_y": 0.0, "min_x": 0.0, "std": 0.95},
    )
    penalty_future_own_landing = RewTerm(
        func=mdp.penalty_future_own_landing_after_hit,
        weight=-60.0,
        params={"max_x": 0.0},
    )
    penalty_actual_own_table_after_hit = RewTerm(
        func=mdp.penalty_own_table_after_paddle_hit,
        weight=-80.0,
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=220.0)
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target,
        weight=120.0,
        params={"target_x": 1.15, "target_y": 0.0, "x_std": 0.7, "y_std": 0.5},
    )
    penalty_hit_low_base_reset = RewTerm(
        func=mdp.penalty_hit_low_base_reset,
        weight=-40.0,
        params={"min_base_z": 0.50},
    )


@configclass
class A3Stage4eEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4eRewardCfg()


@configclass
class A3Stage4eEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4eRewardCfg()


@configclass
class A3Stage4fRewardCfg(A3Stage4dRewardCfg):
    reward_post_hit_ballistic_landing_target = RewTerm(
        func=mdp.reward_post_hit_ballistic_landing_target,
        weight=60.0,
        params={
            "table_z": 0.78,
            "target_x": 1.15,
            "target_y": 0.0,
            "x_std": 0.85,
            "y_std": 0.55,
            "min_vx": 0.1,
            "min_x": 0.0,
            "max_x": 2.7,
            "max_abs_y": 0.9,
            "max_t_land": 1.4,
        },
    )


@configclass
class A3Stage4fEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4fRewardCfg()


@configclass
class A3Stage4fEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4fRewardCfg()


@configclass
class A3Stage4gRewardCfg(A3Stage4dRewardCfg):
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=210.0,
        params={
            "vx_target": 3.4,
            "vz_target": 1.4,
            "z_target": 1.08,
            "z_std": 0.35,
            "min_vx": 0.1,
            "max_t_net": 1.2,
            "t_std": 0.7,
            "vx_weight": 0.60,
            "vz_weight": 0.15,
            "z_weight": 0.20,
            "t_weight": 0.05,
        },
    )
    reward_hit_net_clearance_progress = RewTerm(
        func=mdp.reward_hit_net_clearance_progress,
        weight=80.0,
        params={
            "min_vx": 0.1,
            "vx_target": 2.7,
            "min_z": 0.76,
            "target_z": 1.05,
            "z_std": 0.40,
            "max_t_net": 1.6,
            "t_std": 0.7,
            "vx_weight": 0.70,
            "time_weight": 0.30,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.35, "z_target": 0.76 + 0.32},
        weight=190.0,
    )
    reward_table_success = RewTerm(func=mdp.reward_table_success, weight=190.0)
    reward_actual_opponent_table_target = RewTerm(
        func=mdp.reward_opponent_table_after_paddle_hit_target,
        weight=90.0,
        params={"target_x": 1.15, "target_y": 0.0, "x_std": 0.7, "y_std": 0.5},
    )
    reward_post_hit_net_progress = RewTerm(
        func=mdp.reward_post_hit_net_progress,
        weight=20.0,
        params={
            "min_vx": 0.1,
            "vx_target": 3.5,
            "vz_target": 1.5,
            "x_start": -1.45,
            "max_reward_x": -1.05,
            "net_x": 0.0,
            "net_z_target": 1.08,
            "min_clear_z": 0.78,
            "z_std": 0.40,
            "max_t_net": 1.3,
            "landing_min_x": -1.5,
            "landing_target_x": 1.15,
            "y_target": 0.0,
            "y_std": 0.60,
            "vy_std": 1.50,
            "vx_weight": 0.18,
            "vz_weight": 0.12,
            "x_weight": 0.0,
            "z_weight": 0.30,
            "landing_weight": 0.22,
            "y_weight": 0.18,
        },
    )
    reward_post_hit_ballistic_landing_target = RewTerm(
        func=mdp.reward_post_hit_ballistic_landing_target,
        weight=25.0,
        params={
            "table_z": 0.78,
            "target_x": 1.15,
            "target_y": 0.0,
            "x_std": 1.00,
            "y_std": 0.65,
            "min_vx": 0.1,
            "min_x": 0.0,
            "max_x": 2.7,
            "max_abs_y": 0.9,
            "max_t_land": 1.3,
        },
    )
    penalty_hit_low_base_reset = RewTerm(
        func=mdp.penalty_hit_low_base_reset,
        weight=-30.0,
        params={"min_base_z": 0.50},
    )
    penalty_post_hit_low_base = RewTerm(
        func=mdp.penalty_post_hit_low_base,
        weight=-12.0,
        params={"min_base_z": 0.54, "std": 0.06, "max_penalty": 1.0},
    )


@configclass
class A3Stage4gEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4gRewardCfg()


@configclass
class A3Stage4gEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4gRewardCfg()


@configclass
class A3Stage4hRewardCfg(A3Stage4gRewardCfg):
    reward_hit_ball_velocity_net = RewTerm(
        func=mdp.reward_hit_ball_velocity_net_target,
        weight=210.0,
        params={
            "vx_target": 3.4,
            "vz_target": 1.0,
            "z_target": 1.05,
            "z_std": 0.30,
            "min_vx": 0.1,
            "max_t_net": 1.2,
            "t_std": 0.7,
            "vx_weight": 0.60,
            "vz_weight": 0.05,
            "z_weight": 0.30,
            "t_weight": 0.05,
        },
    )
    reward_future_pass_net = RewTerm(
        func=mdp.reward_future_pass_net,
        params={"std_h": 0.28, "z_target": 0.76 + 0.27},
        weight=220.0,
    )
    reward_post_hit_ballistic_landing_target = RewTerm(
        func=mdp.reward_post_hit_ballistic_landing_target,
        weight=15.0,
        params={
            "table_z": 0.78,
            "target_x": 1.15,
            "target_y": 0.0,
            "x_std": 1.00,
            "y_std": 0.65,
            "min_vx": 0.1,
            "min_x": 0.0,
            "max_x": 2.7,
            "max_abs_y": 0.9,
            "max_t_land": 1.3,
        },
    )
    penalty_post_hit_low_base = RewTerm(
        func=mdp.penalty_post_hit_low_base,
        weight=-25.0,
        params={"min_base_z": 0.54, "std": 0.06, "max_penalty": 1.0},
    )
    penalty_post_hit_trajectory_excess = RewTerm(
        func=mdp.penalty_post_hit_trajectory_excess,
        weight=-18.0,
        params={
            "min_vx": 0.1,
            "net_x": 0.0,
            "max_z_at_net": 1.30,
            "z_std": 0.35,
            "vy_limit": 1.20,
            "vy_std": 1.50,
            "max_t_net": 1.40,
            "max_reward_x": -0.95,
            "z_weight": 0.55,
            "vy_weight": 0.45,
        },
    )


@configclass
class A3Stage4hEnvCfg(A3TableTennisEnvCfg):
    reward = A3Stage4hRewardCfg()


@configclass
class A3Stage4hEvalEnvCfg(A3TT_EvalEnvCfg):
    reward = A3Stage4hRewardCfg()


@configclass
class A3TableTennisAgentCfg(TTAgentCfg):
    experiment_name: str = "a3_table_tennis"
    logger = "tensorboard"
    save_interval = 250
    max_iterations = 10000
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=A3_INITIAL_POLICY_NOISE_STD,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    predictor = {
        "history_len": 5,
        "traj_max_len": 128,
        "hidden_sizes": [64, 64],
        "lr": 0.5e-3,
        "epochs_per_update": 1,
        "batch_size": 1024,
        "train_until_iters": 20,
    }


@configclass
class A3StableReturnAgentCfg(A3TableTennisAgentCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class A3Stage4bAgentCfg(A3StableReturnAgentCfg):
    pass


@configclass
class A3Stage4cAgentCfg(A3TableTennisAgentCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class A3Stage4dAgentCfg(A3TableTennisAgentCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00005,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class A3Stage5ReadyAgentCfg(A3Stage4dAgentCfg):
    pass


@configclass
class A3Stage1BalanceMoveAgentCfg(A3Stage4dAgentCfg):
    experiment_name: str = "a3_table_tennis_stage1"
    save_interval = 10
    max_iterations = 5000
    init_at_random_ep_len = False
    zero_actor_output: bool = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.045,
        noise_std_type="log",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.25,
        use_clipped_value_loss=True,
        clip_param=0.05,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=8,
        learning_rate=1.5e-5,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    algorithm.mean_action_l2_coef = 1.0
    algorithm.deterministic_actions = True
    algorithm.freeze_actor = True


@configclass
class A3Stage1StandAgentCfg(A3Stage1BalanceMoveAgentCfg):
    experiment_name: str = "a3_table_tennis_stage1_stand"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.008,
        noise_std_type="log",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.20,
        use_clipped_value_loss=True,
        clip_param=0.070,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    algorithm.mean_action_l2_coef = 0.05
    algorithm.deterministic_actions = False
    algorithm.freeze_actor = False


@configclass
class A3Stage1RecoveryStandAgentCfg(A3Stage1StandAgentCfg):
    experiment_name: str = "a3_table_tennis_stage1_recovery_stand"
    max_iterations = 1200
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.002,
        noise_std_type="log",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.20,
        use_clipped_value_loss=True,
        clip_param=0.040,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.0010,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    algorithm.mean_action_l2_coef = 0.08
    algorithm.deterministic_actions = False
    algorithm.freeze_actor = False


@configclass
class A3Stage1LateralMoveAgentCfg(A3Stage1StandAgentCfg):
    experiment_name: str = "a3_table_tennis_stage1_lateral_move"
    max_iterations = 2000
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.001,
        noise_std_type="log",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.20,
        use_clipped_value_loss=True,
        clip_param=0.050,
        entropy_coef=0.0,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-6,
        schedule="fixed",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.001,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=True,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    algorithm.mean_action_l2_coef = 0.0
    algorithm.deterministic_actions = False
    algorithm.freeze_actor = False


@configclass
class A3Stage5bAgentCfg(A3Stage4dAgentCfg):
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.16,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00012,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class A3Stage5cAgentCfg(A3Stage4dAgentCfg):
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.12,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00008,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.2e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class A3Stage5dAgentCfg(A3Stage5cAgentCfg):
    pass


@configclass
class A3Stage5eAgentCfg(A3Stage5cAgentCfg):
    pass


@configclass
class A3Stage5fAgentCfg(A3Stage5cAgentCfg):
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.13,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 512, 128],
        critic_hidden_dims=[512, 512, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00010,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.2e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )


@configclass
class A3Stage5gAgentCfg(A3Stage5fAgentCfg):
    pass


@configclass
class A3Stage5gFixedBallAgentCfg(A3Stage5gAgentCfg):
    pass


@configclass
class A3Stage5gWideAgentCfg(A3Stage5gAgentCfg):
    pass


@configclass
class A3Stage5hHitQualityAgentCfg(A3Stage5gWideAgentCfg):
    pass


@configclass
class A3Stage5iStableHitQualityAgentCfg(A3Stage5hHitQualityAgentCfg):
    pass


@configclass
class A3Stage4eAgentCfg(A3Stage4dAgentCfg):
    pass


@configclass
class A3Stage4fAgentCfg(A3Stage4dAgentCfg):
    pass


@configclass
class A3Stage4gAgentCfg(A3Stage4dAgentCfg):
    pass


@configclass
class A3Stage4hAgentCfg(A3Stage4dAgentCfg):
    pass
