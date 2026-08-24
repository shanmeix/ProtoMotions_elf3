# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-contract tests for the ELF3 BXI Steering Stage 1.5 overlay."""

import argparse
from pathlib import Path

import pytest
import yaml

from examples.experiments.steering import mlp_bxi
from examples.experiments.steering import mlp_bxi_stage1_5
from protomotions.tests.test_steering_mlp_bxi_config import _RobotConfig


def test_stage1_5_only_enables_full_circle_random_facing():
    stage1_env_cfg = mlp_bxi.env_config(_RobotConfig(), argparse.Namespace())
    stage1_5_env_cfg = mlp_bxi_stage1_5.env_config(
        _RobotConfig(), argparse.Namespace()
    )

    assert (
        stage1_env_cfg.control_components["steering"].enable_rand_facing
        is False
    )
    stage1_5_steering = stage1_5_env_cfg.control_components["steering"]
    assert stage1_5_steering.enable_rand_facing is True
    assert stage1_5_steering.tar_speed_min == pytest.approx(0.0)
    assert stage1_5_steering.tar_speed_max == pytest.approx(1.5)

    for factory_name in (
        "terrain_config",
        "scene_lib_config",
        "motion_lib_config",
        "agent_config",
        "configure_robot_and_simulator",
        "apply_inference_overrides",
    ):
        assert getattr(mlp_bxi_stage1_5, factory_name) is getattr(
            mlp_bxi, factory_name
        )


def test_stage1_5_motion_set_adds_stand_back_and_preserves_stage1_ratios():
    repo_root = Path(__file__).resolve().parents[2]
    motion_yaml = (
        repo_root
        / "data/yaml_files/elf3_bxi_tienkung_steering_stage1_5.yaml"
    )
    entries = yaml.safe_load(motion_yaml.read_text())["motions"]

    assert [entry["file"] for entry in entries] == [
        "../motions/elf3_bxi/tienkung_locomotion/walk/walk.motion",
        "../motions/elf3_bxi/tienkung_locomotion/amp/stand.motion",
        "../motions/elf3_bxi/tienkung_locomotion/amp/walk_around.motion",
        "../motions/elf3_bxi/tienkung_locomotion/amp/walk_left.motion",
        "../motions/elf3_bxi/tienkung_locomotion/amp/walk_right.motion",
        "../motions/elf3_bxi/tienkung_locomotion/amp/stand_back.motion",
    ]
    weights = [entry["weight"] for entry in entries]
    assert weights == pytest.approx([0.40, 0.08, 0.12, 0.10, 0.10, 0.20])
    assert sum(weights) == pytest.approx(1.0)
    assert all(
        (motion_yaml.parent / entry["file"]).is_file() for entry in entries
    )
