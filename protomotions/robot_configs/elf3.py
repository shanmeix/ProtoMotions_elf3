# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BXI ELF3 robot configuration for ProtoMotions."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from protomotions.components.pose_lib import ControlInfo
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.isaacgym.config import (
    IsaacGymPhysXParams,
    IsaacGymSimParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabPhysXParams,
    IsaacLabSimParams,
)
from protomotions.simulator.newton.config import NewtonSimParams


# Effective armatures inferred from elf3_mjlab's two-stage motor model. At a
# 10 Hz natural frequency, they reproduce the current TienKung-Lab gain table.
ARMATURE_BXI50 = 0.004241984858849
ARMATURE_BXI70 = 0.013735116737364
ARMATURE_BXI85 = 0.044688010203344

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


# Waist pitch/roll and ankle pitch/roll are parallel mechanisms. The scale
# factors preserve the current TienKung-Lab ELF3LITE_CFG gains, including its
# asymmetric waist-roll and ankle-roll transmissions.
ARMATURE_WAIST_Y = 2.0 * ARMATURE_BXI70
ARMATURE_WAIST_X = 3.0 * ARMATURE_BXI70
ARMATURE_ANKLE_Y = 2.0 * ARMATURE_BXI50
ARMATURE_ANKLE_X = 1.3 * ARMATURE_BXI50


DEFAULT_JOINT_POS = {
    ".*_hip_y_joint": -0.3,
    ".*_knee_y_joint": 0.6,
    ".*_ankle_y_joint": -0.3,
    ".*_shoulder_y_joint": 0.2,
    ".*_elbow_y_joint": 0.6,
    "l_shoulder_x_joint": 0.2,
    "r_shoulder_x_joint": -0.2,
}


@dataclass
class Elf3RobotConfig(RobotConfig):
    """Configuration for the 29-DOF BXI ELF3 humanoid."""

    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["l_ankle_x_link"],
            "all_right_foot_bodies": ["r_ankle_x_link"],
            "all_left_hand_bodies": ["l_wrist_z_link"],
            "all_right_hand_bodies": ["r_wrist_z_link"],
            # ELF3-lite has no separate head rigid body; the head mass and
            # geometry are part of the root torso.
            "head_body_name": ["torso_link"],
            "torso_body_name": ["torso_link"],
        }
    )

    # Sparse body set from elf3_mjlab's BeyondMimic tracking task.
    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso_link",
            "l_hip_x_link",
            "l_knee_y_link",
            "l_ankle_x_link",
            "r_hip_x_link",
            "r_knee_y_link",
            "r_ankle_x_link",
            "waist_z_link",
            "l_shoulder_x_link",
            "l_elbow_y_link",
            "l_wrist_z_link",
            "r_shoulder_x_link",
            "r_elbow_y_link",
            "r_wrist_z_link",
        ]
    )

    default_root_height: float = 1.05
    default_dof_pos: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_POS)
    )
    anchor_body_name: str = "torso_link"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/elf3.xml",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "waist_y_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_WAIST_Y),
                    damping=_damping(ARMATURE_WAIST_Y),
                    effort_limit=100.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_WAIST_Y,
                ),
                "waist_x_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_WAIST_X),
                    damping=_damping(ARMATURE_WAIST_X),
                    effort_limit=100.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_WAIST_X,
                ),
                "waist_z_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI85),
                    damping=_damping(ARMATURE_BXI85),
                    effort_limit=100.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI85,
                ),
                ".*_hip_(y|x)_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI85),
                    damping=_damping(ARMATURE_BXI85),
                    effort_limit=100.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI85,
                ),
                ".*_hip_z_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI70),
                    damping=_damping(ARMATURE_BXI70),
                    effort_limit=50.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI70,
                ),
                ".*_knee_y_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI85),
                    damping=_damping(ARMATURE_BXI85),
                    effort_limit=150.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI85,
                ),
                ".*_ankle_y_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_ANKLE_Y),
                    damping=_damping(ARMATURE_ANKLE_Y),
                    effort_limit=50.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_ANKLE_Y,
                ),
                ".*_ankle_x_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_ANKLE_X),
                    damping=_damping(ARMATURE_ANKLE_X),
                    effort_limit=20.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_ANKLE_X,
                ),
                ".*_shoulder_(y|x)_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI70),
                    damping=_damping(ARMATURE_BXI70),
                    effort_limit=50.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI70,
                ),
                ".*_shoulder_z_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI50),
                    damping=_damping(ARMATURE_BXI50),
                    effort_limit=25.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI50,
                ),
                ".*_elbow_y_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI70),
                    damping=_damping(ARMATURE_BXI70),
                    effort_limit=50.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI70,
                ),
                ".*_wrist_(x|y|z)_joint": ControlInfo(
                    stiffness=_stiffness(ARMATURE_BXI50),
                    damping=_damping(ARMATURE_BXI50),
                    effort_limit=25.0,
                    velocity_limit=20.0,
                    armature=ARMATURE_BXI50,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=100,
                decimation=2,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(fps=100, decimation=2, substeps=2),
            newton=NewtonSimParams(fps=200, decimation=4),
        )
    )
