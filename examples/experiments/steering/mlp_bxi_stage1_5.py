# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exploratory full-circle-facing ELF3 BXI walking experiment.

Stage 1.5 intentionally keeps the Stage-1 speed range, observations, action
mapping, robustness setup, and 200 Hz / 4-step timing. Its only task change is
to sample the target facing direction independently over the full circle. The
matching motion set adds ``stand_back`` to the Stage-1 walking references; it
does not contain dedicated lateral or diagonal walking clips, so this
experiment deliberately measures how well the policy extrapolates there.
"""

import argparse

from examples.experiments.steering import mlp_bxi as _base
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig


BXI_ENABLE_RAND_FACING = True

# Stage 1.5 changes only the Steering facing-command distribution.
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
agent_config = _base.agent_config
configure_robot_and_simulator = _base.configure_robot_and_simulator
apply_inference_overrides = _base.apply_inference_overrides


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Enable independent full-circle facing on top of the BXI Stage-1 task."""

    env_cfg = _base.env_config(robot_cfg, args)
    env_cfg.control_components["steering"].enable_rand_facing = (
        BXI_ENABLE_RAND_FACING
    )
    return env_cfg


__all__ = (
    "BXI_ENABLE_RAND_FACING",
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "agent_config",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
)
