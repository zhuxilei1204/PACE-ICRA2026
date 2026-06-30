from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import legged_lab.mdp as mdp
from legged_lab.assets.a3 import A3_T2D5_PINGPANG_CFG
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
