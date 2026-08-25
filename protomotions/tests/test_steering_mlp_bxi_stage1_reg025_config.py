# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the fresh ELF3 BXI Steering Stage1 reg025 run."""

import argparse

import pytest
import torch

from examples.experiments.steering import mlp_bxi
from examples.experiments.steering import mlp_bxi_stage1_reg025
from protomotions.tests.test_steering_mlp_bxi_config import _RobotConfig


def test_stage1_reg025_locks_quarter_scale_rewards_and_stage1_commands():
    env_cfg = mlp_bxi_stage1_reg025.env_config(
        _RobotConfig(), argparse.Namespace()
    )

    assert env_cfg.action_config["action_scale"] == pytest.approx(
        torch.tensor([0.5])
    )
    assert set(mlp_bxi_stage1_reg025.BXI_REQUIRED_REWARD_COMPONENTS).issubset(
        env_cfg.reward_components
    )
    steering_cfg = env_cfg.control_components["steering"]
    assert steering_cfg.enable_rand_facing is False
    assert steering_cfg.tar_speed_min == pytest.approx(0.0)
    assert steering_cfg.tar_speed_max == pytest.approx(1.5)

    for factory_name in (
        "terrain_config",
        "scene_lib_config",
        "motion_lib_config",
        "agent_config",
        "configure_robot_and_simulator",
        "apply_inference_overrides",
    ):
        assert getattr(mlp_bxi_stage1_reg025, factory_name) is getattr(
            mlp_bxi, factory_name
        )


def test_stage1_reg025_validation_rejects_contract_drift():
    robot_cfg = _RobotConfig()
    env_cfg = mlp_bxi.env_config(robot_cfg, argparse.Namespace())

    env_cfg.action_config["action_scale"] = torch.tensor([2.0])
    with pytest.raises(ValueError, match="0.25"):
        mlp_bxi_stage1_reg025._validate_reg025_contract(robot_cfg, env_cfg)

    env_cfg = mlp_bxi.env_config(robot_cfg, argparse.Namespace())
    del env_cfg.reward_components["pd_target_limits"]
    with pytest.raises(ValueError, match="pd_target_limits"):
        mlp_bxi_stage1_reg025._validate_reg025_contract(robot_cfg, env_cfg)

    env_cfg = mlp_bxi.env_config(robot_cfg, argparse.Namespace())
    env_cfg.control_components["steering"].enable_rand_facing = True
    with pytest.raises(ValueError, match="facing aligned"):
        mlp_bxi_stage1_reg025._validate_reg025_contract(robot_cfg, env_cfg)
