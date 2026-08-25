# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF3 BXI Steering Stage 1 with quarter-effort actions and regularization.

This named overlay is intentionally separate from the original Stage-1 run so
``train_agent.py`` creates a fresh resolved configuration instead of resuming
the pre-regularization experiment. It retains the Stage-1 command curriculum,
TienKung walking references, observation contract, robustness setup, and
200 Hz / 4-step timing from :mod:`mlp_bxi`.

The validation below locks the new training contract:

* BM PD targets use ``0.25 * effort_limit / stiffness`` per joint;
* facing remains aligned with travel direction; and
* all seven BXI safety/stance reward components are present.
"""

import argparse

import torch

from examples.experiments.steering import mlp_bxi as _base
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


BXI_ACTION_EFFORT_FRACTION = 0.25
BXI_REQUIRED_REWARD_COMPONENTS = (
    "joint_pos_limits",
    "pd_target_limits",
    "ankle_action",
    "ankle_torque",
    "action_rate",
    "feet_y_distance",
    "zero_command_joint_deviation",
)


terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
configure_robot_and_simulator = _base.configure_robot_and_simulator
apply_inference_overrides = _base.apply_inference_overrides


def _validate_reg025_contract(
    robot_cfg: RobotConfig,
    env_cfg: EnvConfig,
) -> None:
    """Fail early if a future base-config change breaks this experiment."""

    joint_names = robot_cfg.kinematic_info.dof_names
    stiffness = torch.tensor(
        [robot_cfg.control.control_info[name].stiffness for name in joint_names]
    )
    effort_limits = torch.tensor(
        [robot_cfg.control.control_info[name].effort_limit for name in joint_names]
    )
    expected_scale = BXI_ACTION_EFFORT_FRACTION * effort_limits / stiffness
    actual_scale = env_cfg.action_config["action_scale"]
    if not torch.allclose(actual_scale, expected_scale):
        raise ValueError(
            "elf3_bxi Stage1 reg025 requires action_scale="
            "0.25 * effort_limit / stiffness"
        )

    missing_rewards = set(BXI_REQUIRED_REWARD_COMPONENTS).difference(
        env_cfg.reward_components
    )
    if missing_rewards:
        raise ValueError(
            "elf3_bxi Stage1 reg025 is missing required rewards: "
            f"{sorted(missing_rewards)}"
        )

    steering_cfg = env_cfg.control_components["steering"]
    if steering_cfg.enable_rand_facing:
        raise ValueError(
            "elf3_bxi Stage1 reg025 requires facing aligned with travel direction"
        )


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build and validate the fresh regularized Stage-1 configuration."""

    env_cfg = _base.env_config(robot_cfg, args)
    _validate_reg025_contract(robot_cfg, env_cfg)
    return env_cfg


__all__ = (
    "BXI_ACTION_EFFORT_FRACTION",
    "BXI_REQUIRED_REWARD_COMPONENTS",
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "agent_config",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
)
