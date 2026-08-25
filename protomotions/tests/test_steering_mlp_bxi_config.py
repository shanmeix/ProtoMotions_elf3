# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-contract tests for the ELF3 BXI Steering overlay."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.experiments.steering import mlp as steering_mlp
from examples.experiments.steering import mlp_bxi
from protomotions.components.terrains.config import CombineMode
from protomotions.envs.action import bm_pd_action
from protomotions.envs.terminations import fall_termination
from protomotions.robot_configs.factory import robot_config as make_robot_config


class _RobotConfig:
    number_of_actions = 1

    def __init__(self, asset_file: str = "mjcf/elf3_bxi.xml"):
        self.asset = SimpleNamespace(asset_file_name=asset_file)
        self.common_naming_to_robot_body_names = {
            "all_left_foot_bodies": ["left_foot"],
            "all_right_foot_bodies": ["right_foot"],
            "torso_body_name": ["torso"],
        }
        self.kinematic_info = SimpleNamespace(
            num_dofs=1,
            hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
            dof_limits_lower=torch.tensor([-1.0]),
            dof_limits_upper=torch.tensor([1.0]),
            dof_names=["l_ankle_x_joint"],
            body_names=["torso", "left_foot", "right_foot"],
        )
        self.control = SimpleNamespace(
            control_info={
                "l_ankle_x_joint": SimpleNamespace(
                    stiffness=10.0,
                    damping=1.0,
                    effort_limit=20.0,
                )
            }
        )
        self.default_dof_pos = torch.tensor([0.25])
        self.contact_bodies = None
        self.non_termination_contact_bodies = "all"
        self.reset_noise = None
        self.updated = []

    def update_fields(self, **updates):
        self.updated.append(updates)
        for name, value in updates.items():
            setattr(self, name, value)

        requested_contacts = updates.get("contact_bodies")
        if requested_contacts == "all":
            self.contact_bodies = list(self.kinematic_info.body_names)
        elif isinstance(requested_contacts, list):
            resolved = []
            for name in requested_contacts:
                resolved.extend(
                    self.common_naming_to_robot_body_names.get(name, [name])
                )
            self.contact_bodies = resolved


def _simulator_config(*, fps: int = 500, decimation: int = 10):
    return SimpleNamespace(
        sim=SimpleNamespace(fps=fps, decimation=decimation),
        default_robot_friction=1.0,
        domain_randomization=None,
    )


def _args():
    return argparse.Namespace(batch_size=32, training_max_steps=1024)


def test_bxi_overlay_reuses_generic_steering_data_factories():
    assert mlp_bxi.scene_lib_config is steering_mlp.scene_lib_config
    assert mlp_bxi.motion_lib_config is steering_mlp.motion_lib_config

    terrain_cfg = mlp_bxi.terrain_config(argparse.Namespace())
    assert terrain_cfg.sim_config.static_friction == pytest.approx(1.0)
    assert terrain_cfg.sim_config.dynamic_friction == pytest.approx(1.0)
    assert terrain_cfg.sim_config.restitution == pytest.approx(0.0)
    assert terrain_cfg.sim_config.combine_mode is CombineMode.MULTIPLY


def test_real_bxi_action_scales_match_tienkung_locomotion_contract():
    robot_cfg = make_robot_config("elf3_bxi")
    env_cfg = mlp_bxi.env_config(robot_cfg, argparse.Namespace())

    expected_scales = torch.tensor(
        [
            0.231,
            0.154,
            0.213,
            0.213,
            0.213,
            0.231,
            0.213,
            0.373,
            0.230,
            0.213,
            0.213,
            0.231,
            0.213,
            0.373,
            0.230,
            0.231,
            0.231,
            0.373,
            0.231,
            0.373,
            0.373,
            0.373,
            0.231,
            0.231,
            0.373,
            0.231,
            0.373,
            0.373,
            0.373,
        ]
    )
    assert torch.allclose(
        env_cfg.action_config["action_scale"],
        expected_scales,
        atol=6e-4,
    )


def test_bxi_env_uses_first_stage_commands_bm_actions_and_fall_termination():
    robot_cfg = _RobotConfig()
    env_cfg = mlp_bxi.env_config(robot_cfg, argparse.Namespace())

    steering_cfg = env_cfg.control_components["steering"]
    assert steering_cfg.tar_speed_min == pytest.approx(0.0)
    assert steering_cfg.tar_speed_max == pytest.approx(1.5)
    assert steering_cfg.enable_rand_facing is False

    action_cfg = env_cfg.action_config
    assert action_cfg["fn"] is bm_pd_action
    assert torch.equal(action_cfg["pd_action_offset"], torch.tensor([0.25]))
    assert torch.equal(action_cfg["action_scale"], torch.tensor([0.5]))

    rewards = env_cfg.reward_components
    assert list(rewards) == [
        "heading_rew",
        "joint_pos_limits",
        "pd_target_limits",
        "ankle_action",
        "ankle_torque",
        "action_rate",
        "feet_y_distance",
        "zero_command_joint_deviation",
    ]
    assert rewards["joint_pos_limits"].get_bindings_dict()["dof_pos"] == (
        "current.dof_pos"
    )
    assert rewards["joint_pos_limits"].static_params["weight"] == pytest.approx(
        -0.2
    )
    assert torch.allclose(
        rewards["joint_pos_limits"].static_params["dof_limits_lower"],
        torch.tensor([-0.9]),
    )
    assert torch.allclose(
        rewards["joint_pos_limits"].static_params["dof_limits_upper"],
        torch.tensor([0.9]),
    )
    assert rewards["pd_target_limits"].get_bindings_dict()["dof_pos"] == (
        "current_processed_action"
    )
    assert rewards["pd_target_limits"].static_params["weight"] == pytest.approx(
        -0.5
    )
    assert rewards["ankle_action"].get_bindings_dict()["action"] == (
        "current_action"
    )
    assert torch.equal(
        rewards["ankle_action"].static_params["joint_indices"],
        torch.tensor([0]),
    )
    assert rewards["ankle_torque"].get_bindings_dict()["dof_forces"] == (
        "current.dof_forces"
    )
    assert rewards["ankle_torque"].static_params["use_torque_squared"] is True
    assert rewards["action_rate"].get_bindings_dict() == {
        "current_action": "current_action",
        "previous_action": "previous_action",
    }
    assert rewards["feet_y_distance"].static_params[
        "target_distance"
    ] == pytest.approx(0.299)
    assert rewards["feet_y_distance"].static_params[
        "left_foot_body_index"
    ] == 1
    assert rewards["feet_y_distance"].static_params[
        "right_foot_body_index"
    ] == 2
    assert rewards["zero_command_joint_deviation"].get_bindings_dict()[
        "tar_face_dir"
    ] == "steering.tar_face_dir"

    fall_cfg = env_cfg.termination_components["fall"]
    assert fall_cfg.compute_func is fall_termination
    assert fall_cfg.static_params["termination_height"] == pytest.approx(0.2)

    observations = env_cfg.observation_components
    assert observations["max_coords_obs"].get_bindings_dict()["body_pos"] == (
        "current.rigid_body_pos"
    )
    assert observations["historical_max_coords_obs"].get_bindings_dict()[
        "historical_rigid_body_pos"
    ] == "historical.rigid_body_pos"
    assert observations[
        "noisy_historical_reduced_coords_obs"
    ].get_bindings_dict()["historical_dof_pos"] == "noisy_historical.dof_pos"
    assert observations["previous_actions"].get_bindings_dict()[
        "historical_actions"
    ] == "historical.actions"
    assert observations["previous_actions"].static_params["history_steps"] == 8
    assert observations["noisy_steering"].get_bindings_dict()["root_rot"] == (
        "noisy.root_rot"
    )

    agent_cfg = mlp_bxi.agent_config(robot_cfg, env_cfg, _args())
    noisy_actor_keys = [
        "noisy_historical_reduced_coords_obs",
        "previous_actions",
        "noisy_steering",
    ]
    assert agent_cfg.model.actor.in_keys == noisy_actor_keys
    assert agent_cfg.model.actor.mu_model.in_keys == noisy_actor_keys
    assert agent_cfg.model.critic.in_keys == [
        "max_coords_obs",
        "steering",
        "historical_max_coords_obs",
    ]
    assert agent_cfg.model.discriminator.in_keys == [
        "historical_max_coords_obs"
    ]
    assert list(agent_cfg.reference_obs_components) == [
        "historical_max_coords_obs"
    ]
    assert agent_cfg.model.in_keys == [
        "noisy_historical_reduced_coords_obs",
        "previous_actions",
        "noisy_steering",
        "max_coords_obs",
        "steering",
        "historical_max_coords_obs",
    ]


def test_bxi_regularization_components_execute_from_runtime_context():
    env_cfg = mlp_bxi.env_config(_RobotConfig(), argparse.Namespace())
    identity_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    rigid_body_pos = torch.zeros(2, 3, 3)
    rigid_body_pos[:, 1, 1] = 0.1495
    rigid_body_pos[:, 2, 1] = -0.1495
    context = SimpleNamespace(
        current=SimpleNamespace(
            dof_pos=torch.zeros(2, 1),
            dof_vel=torch.zeros(2, 1),
            dof_forces=torch.ones(2, 1),
            rigid_body_pos=rigid_body_pos,
            anchor_rot=identity_rot,
        ),
        current_action=torch.full((2, 1), 0.2),
        previous_action=torch.full((2, 1), 0.1),
        current_processed_action=torch.full((2, 1), 0.95),
        steering=SimpleNamespace(
            tar_dir=torch.tensor([[1.0, 0.0]] * 2),
            tar_speed=torch.zeros(2),
            tar_face_dir=torch.tensor([[1.0, 0.0]] * 2),
        ),
    )

    outputs = {
        name: component.compute(context)
        for name, component in env_cfg.reward_components.items()
        if name != "heading_rew"
    }
    assert set(outputs) == {
        "joint_pos_limits",
        "pd_target_limits",
        "ankle_action",
        "ankle_torque",
        "action_rate",
        "feet_y_distance",
        "zero_command_joint_deviation",
    }
    assert all(value.shape == (2,) for value in outputs.values())
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert torch.allclose(outputs["joint_pos_limits"], torch.zeros(2))
    assert torch.allclose(outputs["pd_target_limits"], torch.full((2,), 0.05))
    assert torch.allclose(
        outputs["zero_command_joint_deviation"], torch.full((2,), 0.25)
    )


def test_bxi_training_inherits_robust_dr_and_enables_full_fall_sensing():
    robot_cfg = _RobotConfig()
    simulator_cfg = _simulator_config()
    timing_cfg = simulator_cfg.sim

    mlp_bxi.configure_robot_and_simulator(
        robot_cfg, simulator_cfg, argparse.Namespace()
    )

    assert simulator_cfg.sim is timing_cfg
    assert (simulator_cfg.sim.fps, simulator_cfg.sim.decimation) == (200, 4)
    assert (
        mlp_bxi.BXI_CONTROL_DECIMATION / mlp_bxi.BXI_PHYSICS_FPS
    ) == pytest.approx(0.02)
    assert simulator_cfg.default_robot_friction == pytest.approx(0.6)
    assert robot_cfg.updated == [
        {"contact_bodies": ["all_left_foot_bodies", "all_right_foot_bodies"]},
        {
            "contact_bodies": "all",
            "non_termination_contact_bodies": ["left_foot", "right_foot"],
        },
    ]
    assert robot_cfg.contact_bodies == ["torso", "left_foot", "right_foot"]
    assert robot_cfg.non_termination_contact_bodies == [
        "left_foot",
        "right_foot",
    ]
    assert robot_cfg.reset_noise.dof_pos_noise == pytest.approx(0.1)

    randomization = simulator_cfg.domain_randomization
    assert randomization.action_noise.action_noise_range == (-0.025, 0.025)
    assert randomization.friction.static_friction_range == (0.3, 1.6)
    assert randomization.friction.dynamic_friction_range == (0.3, 1.2)
    assert randomization.center_of_mass.body_names == ["torso"]
    assert randomization.observation_noise.dof_vel_noise == pytest.approx(0.5)
    assert randomization.push.push_interval_range == (1.0, 3.0)


def test_bxi_training_rejects_wrong_asset_before_overriding_timing():
    simulator_cfg = _simulator_config()

    with pytest.raises(ValueError, match="requires --robot-name elf3_bxi"):
        mlp_bxi.configure_robot_and_simulator(
            _RobotConfig("mjcf/elf3.xml"),
            simulator_cfg,
            argparse.Namespace(),
        )

    assert (simulator_cfg.sim.fps, simulator_cfg.sim.decimation) == (500, 10)


@pytest.mark.parametrize(("fps", "decimation"), [(500, 10), (200, 4)])
def test_bxi_inference_removes_dr_termination_and_uses_nominal_material(
    fps, decimation
):
    robot_cfg = _RobotConfig()
    robot_cfg.contact_bodies = list(robot_cfg.kinematic_info.body_names)
    simulator_cfg = _simulator_config(fps=fps, decimation=decimation)
    simulator_cfg.domain_randomization = object()
    terrain_cfg = mlp_bxi.terrain_config(argparse.Namespace())
    env_cfg = SimpleNamespace(
        termination_components={"fall": object()},
        max_episode_length=300,
        observation_components={
            "noisy_historical_reduced_coords_obs": object(),
            "noisy_steering": object(),
            "previous_actions": object(),
        },
    )
    agent_cfg = SimpleNamespace(
        amp_parameters=SimpleNamespace(discriminator_reward_threshold=0.02)
    )

    mlp_bxi.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        None,
        None,
        argparse.Namespace(),
    )

    assert env_cfg.termination_components == {}
    assert env_cfg.max_episode_length == 1_000_000
    assert env_cfg.observation_components[
        "noisy_historical_reduced_coords_obs"
    ].get_bindings_dict()["historical_dof_pos"] == "historical.dof_pos"
    assert env_cfg.observation_components["noisy_steering"].get_bindings_dict()[
        "root_rot"
    ] == "current.root_rot"
    assert "previous_actions" in env_cfg.observation_components
    assert agent_cfg.amp_parameters.discriminator_reward_threshold == pytest.approx(
        0.0
    )
    assert simulator_cfg.domain_randomization is None
    assert simulator_cfg.default_robot_friction == pytest.approx(0.6)
    assert robot_cfg.contact_bodies is None
    assert terrain_cfg.sim_config.static_friction == pytest.approx(0.6)
    assert terrain_cfg.sim_config.dynamic_friction == pytest.approx(0.6)
    assert terrain_cfg.sim_config.restitution == pytest.approx(0.0)
    assert terrain_cfg.sim_config.combine_mode is CombineMode.AVERAGE


@pytest.mark.parametrize(
    ("asset_file", "fps", "decimation", "message"),
    [
        ("mjcf/elf3.xml", 500, 10, "requires --robot-name elf3_bxi"),
        ("mjcf/elf3_bxi.xml", 200, 5, "requires control_dt=0.02"),
        ("mjcf/elf3_bxi.xml", 0, 4, "timing must be positive"),
    ],
)
def test_bxi_inference_rejects_wrong_asset_or_control_period(
    asset_file, fps, decimation, message
):
    with pytest.raises(ValueError, match=message):
        mlp_bxi.apply_inference_overrides(
            _RobotConfig(asset_file),
            _simulator_config(fps=fps, decimation=decimation),
            None,
            None,
            None,
            None,
            None,
            argparse.Namespace(),
        )
