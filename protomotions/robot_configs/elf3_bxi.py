# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF3 configuration aligned with the BXI MuJoCo deployment model."""

from dataclasses import dataclass, field

from protomotions.robot_configs.base import (
    ControlConfig,
    RobotAssetConfig,
    SimulatorParams,
)
from protomotions.robot_configs.elf3 import Elf3RobotConfig
from protomotions.simulator.mujoco.config import MujocoSimParams


def _elf3_field_default(field_name: str):
    """Create a fresh value using an ``Elf3RobotConfig`` field factory."""

    return Elf3RobotConfig.__dataclass_fields__[field_name].default_factory()


def _bxi_asset_config() -> RobotAssetConfig:
    asset = _elf3_field_default("asset")
    asset.asset_file_name = "mjcf/elf3_bxi.xml"
    return asset


def _bxi_control_config() -> ControlConfig:
    # Start from ELF3 so stiffness, damping, armature, velocity limits, and the
    # regex-to-joint mapping stay identical. Only the active BXI motor limits
    # that differ from the original asset are changed here.
    control = _elf3_field_default("control")
    control.override_control_info["waist_z_joint"].effort_limit = 150.0
    control.override_control_info[".*_hip_(y|x)_joint"].effort_limit = 150.0
    return control


def _bxi_simulation_params() -> SimulatorParams:
    # Preserve the ELF3 settings for the other backends. BXI runs physics at
    # 500 Hz and applies a policy action every ten substeps (50 Hz / 20 ms).
    params = _elf3_field_default("simulation_params")
    params.isaaclab.fps = 500
    params.isaaclab.decimation = 10
    params.mujoco = MujocoSimParams(fps=500, decimation=10)
    return params


@dataclass
class Elf3BxiRobotConfig(Elf3RobotConfig):
    """29-DOF ELF3 using the physical parameters of the BXI simulator."""

    asset: RobotAssetConfig = field(default_factory=_bxi_asset_config)
    control: ControlConfig = field(default_factory=_bxi_control_config)
    simulation_params: SimulatorParams = field(
        default_factory=_bxi_simulation_params
    )
