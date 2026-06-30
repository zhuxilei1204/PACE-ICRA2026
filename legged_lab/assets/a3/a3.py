import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


A3_T2D5_URDF_PATH = f"{ISAAC_ASSET_DIR}/a3/t2d5/a3_t2d5/urdf/model.urdf"
A3_T2D5_PINGPANG_URDF_PATH = f"{ISAAC_ASSET_DIR}/a3/t2d5_pingpang/a3_t2d5_pingpang/urdf/model.urdf"

A3_ARMATURE = {
    # J_output = J_motor * gear_ratio^2. Values come from the A3 actuator table.
    "pfp_110_75": 0.12034028684,
    "pfp_93_65": 0.06646569890856839,
    "pfp_78_58": 0.012083368706200002,
    "pfp_59_60": 0.0049673513029504,
    "pfp_41_48": 0.0008100893337749999,
    "ankle_pitch_parallel": 0.0644406053101646,
    "ankle_roll_parallel": 0.020126300584420845,
    "waist_pitch_parallel": 0.08820859155526,
    "waist_roll_parallel": 0.014620876134502,
}

A3_RATED_SPEED = {
    "pfp_110_75": 14.660765716752367,
    "pfp_93_65": 12.042771838760874,
    "pfp_78_58": 13.613568165555769,
    "pfp_59_60": 14.660765716752367,
    "pfp_41_48": 15.707963267948966,
}

A3_STABLE_STANDING_ROOT_POS = (-1.6, 0.0, 1.055)

A3_STABLE_STANDING_JOINT_POS = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": -0.04,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    # Measured right-arm blend, with wrist pose matching the T1 paddle frame.
    "right_shoulder_pitch_joint": 0.1449383158874511,
    "right_shoulder_roll_joint": -0.053864232177734285,
    "right_shoulder_yaw_joint": 0.004107922210693449,
    "right_elbow_joint": 0.4118487550354004,
    "right_wrist_roll_joint": -0.05,
    "right_wrist_pitch_joint": -0.33,
    "right_wrist_yaw_joint": -1.1,
    "left_hip_pitch_joint": -0.05,
    "left_hip_roll_joint": 0.16,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.50,
    "left_ankle_pitch_joint": -0.22,
    "left_ankle_roll_joint": -0.072,
    "right_hip_pitch_joint": -0.05,
    "right_hip_roll_joint": -0.16,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.50,
    "right_ankle_pitch_joint": -0.22,
    "right_ankle_roll_joint": 0.072,
}


A3_PINGPONG_READY_JOINT_POS = A3_STABLE_STANDING_JOINT_POS.copy()
A3_PINGPONG_READY_JOINT_POS.update(
    {
        # Candidate 69 from A3 standing search with the current paddle-ready arm.
        "left_hip_pitch_joint": -0.18,
        "left_hip_roll_joint": -0.16,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.50,
        "left_ankle_pitch_joint": -0.26,
        "left_ankle_roll_joint": 0.072,
        "right_hip_pitch_joint": -0.18,
        "right_hip_roll_joint": 0.16,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.50,
        "right_ankle_pitch_joint": -0.26,
        "right_ankle_roll_joint": -0.072,
        # A3 pingpong wrist candidate rollneg/pitch125: improves early hit vx direction.
        "right_wrist_roll_joint": -0.25,
        "right_wrist_pitch_joint": -1.25,
        "right_wrist_yaw_joint": -1.40,
    }
)

A3_PINGPONG_DAMPING_SCALES = {
    "waist": 2.0,
    "legs": 3.0,
    "feet": 3.0,
}


def _scale_actuator_value(value, scale: float):
    if scale == 1.0:
        return value
    if isinstance(value, dict):
        return {key: float(item) * scale for key, item in value.items()}
    if isinstance(value, (float, int)):
        return float(value) * scale
    return value


def _with_pingpong_damping(actuators):
    return {
        name: actuator.replace(
            damping=_scale_actuator_value(actuator.damping, A3_PINGPONG_DAMPING_SCALES.get(name, 1.0))
        )
        for name, actuator in actuators.items()
    }


A3_T2D5_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=A3_T2D5_URDF_PATH,
        usd_dir=f"{ISAAC_ASSET_DIR}/a3/t2d5/generated",
        usd_file_name="A3_T2D5.usd",
        fix_base=False,
        root_link_name="pelvis_link",
        merge_fixed_joints=False,
        force_usd_conversion=False,
        make_instanceable=True,
        activate_contact_sensors=True,
        collision_from_visuals=False,
        collider_type="convex_hull",
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
            target_type="position",
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=A3_STABLE_STANDING_ROOT_POS,
        joint_pos=A3_STABLE_STANDING_JOINT_POS.copy(),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 220.0,
                "waist_roll_joint": 46.0,
                "waist_pitch_joint": 115.0,
            },
            velocity_limit={
                "waist_yaw_joint": A3_RATED_SPEED["pfp_93_65"],
                "waist_roll_joint": 22.7,
                "waist_pitch_joint": 9.25,
            },
            stiffness=80.0,
            damping=8.0,
            armature={
                "waist_yaw_joint": A3_ARMATURE["pfp_93_65"],
                "waist_roll_joint": A3_ARMATURE["waist_roll_parallel"],
                "waist_pitch_joint": A3_ARMATURE["waist_pitch_parallel"],
            },
            min_delay=0,
            max_delay=3,
        ),
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*hip_pitch_joint",
                ".*hip_roll_joint",
                ".*hip_yaw_joint",
                ".*knee_joint",
            ],
            effort_limit={
                ".*hip_pitch_joint": 220.0,
                ".*hip_roll_joint": 220.0,
                ".*hip_yaw_joint": 220.0,
                ".*knee_joint": 320.0,
            },
            velocity_limit={
                ".*hip_pitch_joint": A3_RATED_SPEED["pfp_93_65"],
                ".*hip_roll_joint": A3_RATED_SPEED["pfp_93_65"],
                ".*hip_yaw_joint": A3_RATED_SPEED["pfp_93_65"],
                ".*knee_joint": A3_RATED_SPEED["pfp_110_75"],
            },
            stiffness=120.0,
            damping=12.0,
            armature={
                ".*hip_pitch_joint": A3_ARMATURE["pfp_93_65"],
                ".*hip_roll_joint": A3_ARMATURE["pfp_93_65"],
                ".*hip_yaw_joint": A3_ARMATURE["pfp_93_65"],
                ".*knee_joint": A3_ARMATURE["pfp_110_75"],
            },
            min_delay=0,
            max_delay=3,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*ankle_pitch_joint",
                ".*ankle_roll_joint",
            ],
            effort_limit={
                ".*ankle_pitch_joint": 118.2,
                ".*ankle_roll_joint": 54.75,
            },
            velocity_limit={
                ".*ankle_pitch_joint": 10.8,
                ".*ankle_roll_joint": 19.37,
            },
            stiffness=60.0,
            damping=12.0,
            armature={
                ".*ankle_pitch_joint": A3_ARMATURE["ankle_pitch_parallel"],
                ".*ankle_roll_joint": A3_ARMATURE["ankle_roll_parallel"],
            },
            min_delay=0,
            max_delay=3,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*shoulder_.*_joint",
                ".*elbow_joint",
            ],
            effort_limit={
                ".*shoulder_pitch_joint": 60.0,
                ".*shoulder_roll_joint": 60.0,
                ".*shoulder_yaw_joint": 36.0,
                ".*elbow_joint": 36.0,
            },
            velocity_limit={
                ".*shoulder_pitch_joint": A3_RATED_SPEED["pfp_78_58"],
                ".*shoulder_roll_joint": A3_RATED_SPEED["pfp_78_58"],
                ".*shoulder_yaw_joint": A3_RATED_SPEED["pfp_59_60"],
                ".*elbow_joint": A3_RATED_SPEED["pfp_59_60"],
            },
            stiffness=35.0,
            damping=4.0,
            armature={
                ".*shoulder_pitch_joint": A3_ARMATURE["pfp_78_58"],
                ".*shoulder_roll_joint": A3_ARMATURE["pfp_78_58"],
                ".*shoulder_yaw_joint": A3_ARMATURE["pfp_59_60"],
                ".*elbow_joint": A3_ARMATURE["pfp_59_60"],
            },
            min_delay=0,
            max_delay=3,
        ),
        "wrists": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*wrist_.*_joint",
            ],
            effort_limit={
                ".*wrist_roll_joint": 36.0,
                ".*wrist_pitch_joint": 6.0,
                ".*wrist_yaw_joint": 6.0,
            },
            velocity_limit={
                ".*wrist_roll_joint": A3_RATED_SPEED["pfp_59_60"],
                ".*wrist_pitch_joint": A3_RATED_SPEED["pfp_41_48"],
                ".*wrist_yaw_joint": A3_RATED_SPEED["pfp_41_48"],
            },
            stiffness=20.0,
            damping=3.0,
            armature={
                ".*wrist_roll_joint": A3_ARMATURE["pfp_59_60"],
                ".*wrist_pitch_joint": A3_ARMATURE["pfp_41_48"],
                ".*wrist_yaw_joint": A3_ARMATURE["pfp_41_48"],
            },
            min_delay=0,
            max_delay=3,
        ),
        "head": DelayedPDActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
            ],
            effort_limit=6.0,
            velocity_limit=A3_RATED_SPEED["pfp_41_48"],
            stiffness=10.0,
            damping=1.0,
            armature=A3_ARMATURE["pfp_41_48"],
            min_delay=0,
            max_delay=3,
        ),
    },
)
"""Configuration for the A3T2.5 humanoid robot."""


A3_T2D5_PINGPANG_CFG = A3_T2D5_CFG.replace(
    spawn=A3_T2D5_CFG.spawn.replace(
        asset_path=A3_T2D5_PINGPANG_URDF_PATH,
        usd_dir=f"{ISAAC_ASSET_DIR}/a3/t2d5_pingpang/generated",
        usd_file_name="A3_T2D5_PINGPANG.usd",
    ),
    init_state=A3_T2D5_CFG.init_state.replace(joint_pos=A3_PINGPONG_READY_JOINT_POS.copy()),
    actuators=_with_pingpong_damping(A3_T2D5_CFG.actuators),
)
"""Configuration for the A3T2.5 humanoid robot with the ping-pong paddle asset."""
