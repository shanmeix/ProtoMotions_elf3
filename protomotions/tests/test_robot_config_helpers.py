# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot config parsing and factory helpers."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import protomotions.components.pose_lib as pose_lib
import protomotions.robot_configs.base as robot_base
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    abstract_names_to_body_names,
)
from protomotions.robot_configs.factory import robot_config


def _names():
    return {
        "all_left_foot_bodies": ["left_foot"],
        "all_right_foot_bodies": ["right_foot"],
        "all_left_hand_bodies": ["left_hand"],
        "all_right_hand_bodies": ["right_hand"],
        "head_body_name": ["head"],
        "torso_body_name": ["torso"],
    }


def _patch_pose_extractors(monkeypatch):
    kinematic_info = SimpleNamespace(
        body_names=["root", "torso", "head", "left_foot", "right_foot"],
        dof_names=["left_knee_joint", "right_hip_joint", "spine_joint"],
        num_dofs=3,
    )
    monkeypatch.setattr(pose_lib, "extract_kinematic_info", lambda mjcf_path: kinematic_info)
    monkeypatch.setattr(
        pose_lib,
        "extract_control_info",
        lambda mjcf_path, override_control_info=None: {"path": mjcf_path, "override": override_control_info},
    )
    return kinematic_info


def _robot(monkeypatch, **kwargs):
    _patch_pose_extractors(monkeypatch)
    params = {
        "asset": RobotAssetConfig(asset_root="/assets", asset_file_name="robot.xml"),
        "common_naming_to_robot_body_names": _names(),
        "semantic_forward_axis_xy": [1.0, 0.0],
    }
    params.update(kwargs)
    return RobotConfig(**params)


def test_control_type_and_asset_validation():
    assert ControlType.from_str("TORQUE") is ControlType.TORQUE
    assert ControlType.from_str("built_in_pd") is ControlType.BUILT_IN_PD
    with pytest.raises(ValueError, match="not a valid ControlType"):
        ControlType.from_str("velocity")

    with pytest.raises(ValueError, match="must be a valid path"):
        RobotAssetConfig(asset_file_name="robot.urdf")


def test_control_config_converts_dict_overrides(monkeypatch):
    class _ControlInfo:
        @staticmethod
        def from_dict(data):
            return SimpleNamespace(converted=data)

    monkeypatch.setattr(robot_base, "ControlInfo", _ControlInfo)

    config = ControlConfig(
        override_control_info={
            "joint": {"stiffness": 10.0},
            "ready": SimpleNamespace(value=1),
        }
    )

    assert config.override_control_info["joint"].converted == {"stiffness": 10.0}
    assert config.override_control_info["ready"].value == 1


def test_robot_config_post_init_resolves_anchor_defaults_and_abstract_body_names(monkeypatch):
    config = _robot(
        monkeypatch,
        anchor_body_name="torso",
        default_dof_pos={".*_knee_joint": 0.7, "spine_.*": -0.2},
        mimic_small_marker_bodies=["root", "head_body_name"],
        contact_bodies="all_left_foot_bodies",
        trackable_bodies_subset="all",
    )

    assert config.anchor_body_index == 1
    assert torch.allclose(config.default_dof_pos, torch.tensor([0.7, 0.0, -0.2]))
    assert config.number_of_actions == 3
    assert config.mimic_small_marker_bodies == ["root", "head"]
    assert config.contact_bodies == ["left_foot"]
    assert config.trackable_bodies_subset == config.kinematic_info.body_names
    assert config.control.control_info["path"] == "/assets/robot.xml"


def test_robot_config_post_init_validates_anchor_and_default_dof_length(monkeypatch):
    with pytest.raises(ValueError, match="anchor_body_name"):
        _robot(monkeypatch, anchor_body_name="missing")

    with pytest.raises(AssertionError, match="default_dof_pos length"):
        _robot(monkeypatch, default_dof_pos=[0.0, 1.0])

    with pytest.raises(AssertionError, match="must contain"):
        _robot(monkeypatch, common_naming_to_robot_body_names={})


def test_robot_config_requires_and_normalizes_semantic_forward_axis(monkeypatch):
    _patch_pose_extractors(monkeypatch)
    params = {
        "asset": RobotAssetConfig(asset_root="/assets", asset_file_name="robot.xml"),
        "common_naming_to_robot_body_names": _names(),
    }

    with pytest.raises(ValueError, match="semantic_forward_axis_xy") as exc_info:
        RobotConfig(**params)
    assert "scripts/identify_robot_facing_axis.py" in str(exc_info.value)
    assert "--view" in str(exc_info.value)

    config = _robot(monkeypatch, semantic_forward_axis_xy=[3.0, 4.0])
    assert config.semantic_forward_axis_xy == pytest.approx((0.6, 0.8))

    with pytest.raises(ValueError, match="non-zero 2D"):
        _robot(monkeypatch, semantic_forward_axis_xy=[0.0, 0.0])


def test_robot_config_update_fields_reprocesses_body_name_aliases(monkeypatch):
    config = _robot(monkeypatch, contact_bodies=None, trackable_bodies_subset=["root"])

    config.update_fields(contact_bodies=["root", "all_right_foot_bodies"])

    assert config.contact_bodies == ["root", "right_foot"]
    with pytest.raises(ValueError, match="has no field"):
        config.update_fields(missing=True)


def test_abstract_names_to_body_names_handles_none_all_lists_and_literals(monkeypatch):
    config = _robot(monkeypatch)

    assert abstract_names_to_body_names(None, config) is None
    assert abstract_names_to_body_names("all", config) == config.kinematic_info.body_names
    assert abstract_names_to_body_names("root", config) == ["root"]
    assert abstract_names_to_body_names("all_left_hand_bodies", config) == ["left_hand"]
    assert abstract_names_to_body_names(["root", "custom"], config) == ["root", "custom"]


def test_robot_config_factory_dispatches_all_robot_names_and_applies_updates(monkeypatch):
    class _FactoryConfig:
        def __init__(self):
            self.updates = {}

        def update_fields(self, **updates):
            self.updates.update(updates)

    for module_name, class_name in [
        ("protomotions.robot_configs.smpl", "SmplRobotConfig"),
        ("protomotions.robot_configs.smplx", "SMPLXRobotConfig"),
        ("protomotions.robot_configs.amp", "AMPRobotConfig"),
        ("protomotions.robot_configs.g1", "G1RobotConfig"),
        ("protomotions.robot_configs.h1_2", "H1_2RobotConfig"),
        ("protomotions.robot_configs.soma23", "Soma23RobotConfig"),
        ("protomotions.robot_configs.elf3", "Elf3RobotConfig"),
        ("protomotions.robot_configs.elf3_bxi", "Elf3BxiRobotConfig"),
    ]:
        module = types.ModuleType(module_name)
        setattr(module, class_name, _FactoryConfig)
        monkeypatch.setitem(sys.modules, module_name, module)

    for name in [
        "smpl",
        "smplx",
        "amp",
        "g1",
        "h1_2",
        "soma23",
        "elf3",
        "elf3_bxi",
    ]:
        config = robot_config(name, trackable_bodies_subset=["root"])
        assert isinstance(config, _FactoryConfig)
        assert config.updates == {"trackable_bodies_subset": ["root"]}

    with pytest.raises(ValueError, match="Invalid robot name"):
        robot_config("unknown")


def test_elf3_robot_config_contract():
    config = robot_config("elf3")
    expected_dof_names = [
        "waist_y_joint",
        "waist_x_joint",
        "waist_z_joint",
        "l_hip_y_joint",
        "l_hip_x_joint",
        "l_hip_z_joint",
        "l_knee_y_joint",
        "l_ankle_y_joint",
        "l_ankle_x_joint",
        "r_hip_y_joint",
        "r_hip_x_joint",
        "r_hip_z_joint",
        "r_knee_y_joint",
        "r_ankle_y_joint",
        "r_ankle_x_joint",
        "l_shoulder_y_joint",
        "l_shoulder_x_joint",
        "l_shoulder_z_joint",
        "l_elbow_y_joint",
        "l_wrist_x_joint",
        "l_wrist_y_joint",
        "l_wrist_z_joint",
        "r_shoulder_y_joint",
        "r_shoulder_x_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
        "r_wrist_x_joint",
        "r_wrist_y_joint",
        "r_wrist_z_joint",
    ]

    assert config.kinematic_info.dof_names == expected_dof_names
    assert config.kinematic_info.num_dofs == config.number_of_actions == 29
    assert config.kinematic_info.num_bodies == 30
    assert config.anchor_body_name == "torso_link"
    assert config.anchor_body_index == 0
    assert config.default_root_height == pytest.approx(1.05)
    assert config.control.control_type is ControlType.BUILT_IN_PD
    assert set(config.control.control_info) == set(expected_dof_names)

    expected_aliases = {
        "all_left_foot_bodies": ["l_ankle_x_link"],
        "all_right_foot_bodies": ["r_ankle_x_link"],
        "all_left_hand_bodies": ["l_wrist_z_link"],
        "all_right_hand_bodies": ["r_wrist_z_link"],
        "head_body_name": ["torso_link"],
        "torso_body_name": ["torso_link"],
    }
    assert config.common_naming_to_robot_body_names == expected_aliases
    assert len(config.trackable_bodies_subset) == 14
    assert len(set(config.trackable_bodies_subset)) == 14
    assert set(config.trackable_bodies_subset) <= set(config.kinematic_info.body_names)

    expected_default_pos = {
        "l_hip_y_joint": -0.3,
        "r_hip_y_joint": -0.3,
        "l_knee_y_joint": 0.6,
        "r_knee_y_joint": 0.6,
        "l_ankle_y_joint": -0.3,
        "r_ankle_y_joint": -0.3,
        "l_shoulder_y_joint": 0.2,
        "r_shoulder_y_joint": 0.2,
        "l_shoulder_x_joint": 0.2,
        "r_shoulder_x_joint": -0.2,
        "l_elbow_y_joint": 0.6,
        "r_elbow_y_joint": 0.6,
    }
    for index, dof_name in enumerate(expected_dof_names):
        expected = expected_default_pos.get(dof_name, 0.0)
        assert config.default_dof_pos[index].item() == pytest.approx(expected)

    for control_info in config.control.control_info.values():
        assert control_info.stiffness is not None and control_info.stiffness > 0
        assert control_info.damping is not None and control_info.damping > 0
        assert control_info.effort_limit is not None and control_info.effort_limit > 0
        assert control_info.velocity_limit is not None and control_info.velocity_limit > 0
        assert control_info.armature is not None and control_info.armature > 0


def test_elf3_bxi_robot_config_contract():
    config = robot_config("elf3_bxi")
    elf3_config = robot_config("elf3")

    assert config.asset.asset_file_name == "mjcf/elf3_bxi.xml"
    assert config.kinematic_info.dof_names == elf3_config.kinematic_info.dof_names
    assert config.kinematic_info.body_names == elf3_config.kinematic_info.body_names
    assert config.kinematic_info.num_dofs == config.number_of_actions == 29
    assert config.kinematic_info.num_bodies == 30
    assert config.anchor_body_name == elf3_config.anchor_body_name == "torso_link"
    assert config.anchor_body_index == elf3_config.anchor_body_index == 0
    assert config.semantic_forward_axis_xy == elf3_config.semantic_forward_axis_xy
    assert config.common_naming_to_robot_body_names == (
        elf3_config.common_naming_to_robot_body_names
    )
    assert config.trackable_bodies_subset == elf3_config.trackable_bodies_subset
    assert torch.equal(config.default_dof_pos, elf3_config.default_dof_pos)

    expected_effort_limits = {
        "waist_y_joint": 100.0,
        "waist_x_joint": 100.0,
        "waist_z_joint": 150.0,
        ".*_hip_(y|x)_joint": 150.0,
        ".*_hip_z_joint": 50.0,
        ".*_knee_y_joint": 150.0,
        ".*_ankle_y_joint": 50.0,
        ".*_ankle_x_joint": 20.0,
        ".*_shoulder_(y|x)_joint": 50.0,
        ".*_shoulder_z_joint": 25.0,
        ".*_elbow_y_joint": 50.0,
        ".*_wrist_(x|y|z)_joint": 25.0,
    }
    assert set(config.control.override_control_info) == set(expected_effort_limits)
    for pattern, expected_limit in expected_effort_limits.items():
        bxi_control = config.control.override_control_info[pattern]
        elf3_control = elf3_config.control.override_control_info[pattern]
        assert bxi_control.effort_limit == pytest.approx(expected_limit)
        assert bxi_control.stiffness == pytest.approx(elf3_control.stiffness)
        assert bxi_control.damping == pytest.approx(elf3_control.damping)
        assert bxi_control.armature == pytest.approx(elf3_control.armature)
        assert bxi_control.velocity_limit == pytest.approx(
            elf3_control.velocity_limit
        )

    assert config.simulation_params.isaacgym == elf3_config.simulation_params.isaacgym
    assert config.simulation_params.genesis == elf3_config.simulation_params.genesis
    assert config.simulation_params.newton == elf3_config.simulation_params.newton
    assert (
        config.simulation_params.isaaclab.physx
        == elf3_config.simulation_params.isaaclab.physx
    )
    assert config.simulation_params.isaaclab.fps == 500
    assert config.simulation_params.isaaclab.decimation == 10
    assert config.simulation_params.mujoco.fps == 500
    assert config.simulation_params.mujoco.decimation == 10
    assert (
        config.simulation_params.isaaclab.decimation
        / config.simulation_params.isaaclab.fps
        == pytest.approx(0.02)
    )
    assert (
        config.simulation_params.mujoco.decimation
        / config.simulation_params.mujoco.fps
        == pytest.approx(0.02)
    )

    # Every dataclass default factory must return independent mutable objects.
    second = robot_config("elf3_bxi")
    assert second.asset is not config.asset
    assert second.control is not config.control
    assert second.control.override_control_info is not (
        config.control.override_control_info
    )
    assert second.simulation_params is not config.simulation_params
    assert second.simulation_params.isaaclab is not config.simulation_params.isaaclab
    assert second.simulation_params.mujoco is not config.simulation_params.mujoco
