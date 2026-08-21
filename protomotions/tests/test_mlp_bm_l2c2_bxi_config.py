# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thin ELF3 BXI BeyondMimic experiment overlay."""

import argparse
from types import SimpleNamespace

import pytest

from examples.experiments.mimic import mlp_bm_l2c2
from examples.experiments.mimic import mlp_bm_l2c2_bxi
from protomotions.components.terrains.config import CombineMode


class _RobotConfig:
    def __init__(self, asset_file="mjcf/elf3_bxi.xml"):
        self.asset = SimpleNamespace(asset_file_name=asset_file)
        self.common_naming_to_robot_body_names = {
            "torso_body_name": ["torso_link"],
        }
        self.updated = []
        self.reset_noise = None

    def update_fields(self, **updates):
        self.updated.append(updates)


def _simulator_config(*, fps=500, decimation=10):
    return SimpleNamespace(
        sim=SimpleNamespace(fps=fps, decimation=decimation),
        default_robot_friction=1.0,
        domain_randomization=None,
    )


def test_bxi_overlay_reuses_the_beyondmimic_task_and_agent_factories():
    assert mlp_bm_l2c2_bxi.terrain_config is mlp_bm_l2c2.terrain_config
    assert mlp_bm_l2c2_bxi.scene_lib_config is mlp_bm_l2c2.scene_lib_config
    assert mlp_bm_l2c2_bxi.motion_lib_config is mlp_bm_l2c2.motion_lib_config
    assert mlp_bm_l2c2_bxi.env_config is mlp_bm_l2c2.env_config
    assert mlp_bm_l2c2_bxi.agent_config is mlp_bm_l2c2.agent_config


def test_bxi_overlay_inherits_domain_randomization_without_overwriting_timing():
    robot_cfg = _RobotConfig()
    simulator_cfg = _simulator_config()
    timing_config = simulator_cfg.sim

    mlp_bm_l2c2_bxi.configure_robot_and_simulator(
        robot_cfg, simulator_cfg, argparse.Namespace()
    )

    assert simulator_cfg.sim is timing_config
    assert (simulator_cfg.sim.fps, simulator_cfg.sim.decimation) == (500, 10)
    assert simulator_cfg.default_robot_friction == pytest.approx(0.6)
    assert robot_cfg.updated == [
        {"contact_bodies": ["all_left_foot_bodies", "all_right_foot_bodies"]}
    ]
    assert robot_cfg.reset_noise.dof_pos_noise == pytest.approx(0.1)

    randomization = simulator_cfg.domain_randomization
    assert randomization.action_noise.action_noise_range == (-0.025, 0.025)
    assert randomization.friction.static_friction_range == (0.3, 1.6)
    assert randomization.friction.dynamic_friction_range == (0.3, 1.2)
    assert randomization.friction.restitution_range == (0.0, 0.5)
    assert randomization.center_of_mass.body_names == ["torso_link"]
    assert randomization.observation_noise.dof_vel_noise == pytest.approx(0.5)
    assert randomization.push.push_interval_range == (1.0, 3.0)


@pytest.mark.parametrize(
    ("asset_file", "fps", "decimation", "message"),
    [
        ("mjcf/elf3.xml", 500, 10, "requires --robot-name elf3_bxi"),
        ("mjcf/elf3_bxi.xml", 200, 4, "robot config"),
    ],
)
def test_bxi_overlay_rejects_the_wrong_robot_layer(
    asset_file, fps, decimation, message
):
    with pytest.raises(ValueError, match=message):
        mlp_bm_l2c2_bxi.configure_robot_and_simulator(
            _RobotConfig(asset_file),
            _simulator_config(fps=fps, decimation=decimation),
            argparse.Namespace(),
        )


def test_bxi_inference_rejects_a_non_bxi_frozen_robot_config():
    with pytest.raises(ValueError, match="requires --robot-name elf3_bxi"):
        mlp_bm_l2c2_bxi.apply_inference_overrides(
            _RobotConfig("mjcf/elf3.xml"),
            _simulator_config(),
            None,
            None,
            None,
            None,
            None,
            argparse.Namespace(),
        )


@pytest.mark.parametrize(("fps", "decimation"), [(500, 10), (200, 4)])
def test_bxi_inference_uses_nominal_friction_and_base_inference_cleanup(
    fps, decimation
):
    robot_cfg = _RobotConfig()
    simulator_cfg = _simulator_config(fps=fps, decimation=decimation)
    simulator_cfg.domain_randomization = object()
    terrain_cfg = mlp_bm_l2c2_bxi.terrain_config(argparse.Namespace())
    env_cfg = SimpleNamespace(
        termination_components={"fall": object()},
        max_episode_length=10,
        motion_manager=SimpleNamespace(
            resample_on_reset=False,
            init_start_prob=0.0,
        ),
        observation_components={
            "noisy_reduced_coords_obs": object(),
            "noisy_mimic_reduced_coords_target_poses": object(),
            "clean_reduced_coords_obs": object(),
            "clean_mimic_reduced_coords_target_poses": object(),
        },
    )

    mlp_bm_l2c2_bxi.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        None,
        terrain_cfg,
        None,
        None,
        argparse.Namespace(),
    )

    assert env_cfg.termination_components == {}
    assert env_cfg.max_episode_length == 1_000_000
    assert "clean_reduced_coords_obs" not in env_cfg.observation_components
    assert simulator_cfg.domain_randomization is None
    assert simulator_cfg.default_robot_friction == pytest.approx(0.6)
    assert terrain_cfg.sim_config.static_friction == pytest.approx(0.6)
    assert terrain_cfg.sim_config.dynamic_friction == pytest.approx(0.6)
    assert terrain_cfg.sim_config.restitution == pytest.approx(0.0)
    assert terrain_cfg.sim_config.combine_mode is CombineMode.AVERAGE


def test_bxi_inference_rejects_the_wrong_control_period():
    with pytest.raises(ValueError, match="requires control_dt=0.02"):
        mlp_bm_l2c2_bxi.apply_inference_overrides(
            _RobotConfig(),
            _simulator_config(fps=200, decimation=5),
            None,
            None,
            None,
            None,
            None,
            argparse.Namespace(),
        )
