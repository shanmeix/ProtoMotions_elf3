# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ELF3 BXI overlay for the AMP Steering locomotion experiment.

The generic Steering experiment remains the single source of truth for the
task observations, reward, and AMP agent.  This module only adds the pieces
that are specific to the BXI locomotion training/deployment contract:

* BXI asset and Steering-local 200 Hz / 4-step training timing validation;
* the deployment-aligned BeyondMimic PD action mapping;
* deployment-observable actor inputs with training-time sensor noise;
* the robust reset/domain-randomization setup already used by BXI mimic;
* a first-stage command distribution covered by the TienKung motion set;
* TienKung-derived joint, ankle, smoothness, and stance regularization; and
* fall termination during training, removed again for inference.
"""

import argparse
import math

import torch

from examples.experiments.mimic import mlp_bm_l2c2 as _robustness
from examples.experiments.mimic import mlp_bm_l2c2_bxi as _bxi
from examples.experiments.steering import mlp as _base
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import (
    CombineMode,
    TerrainConfig,
    TerrainSimConfig,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


BXI_ASSET_FILE = _bxi.BXI_ASSET_FILE
BXI_NOMINAL_FRICTION = _bxi.BXI_NOMINAL_FRICTION
BXI_PHYSICS_FPS = 200
BXI_CONTROL_DECIMATION = 4
BXI_CONTROL_DT = BXI_CONTROL_DECIMATION / BXI_PHYSICS_FPS

# First-stage commands stay inside the stand/walk/turn portion of the converted
# TienKung data.  A later curriculum can raise the maximum when run clips are
# deliberately introduced.
BXI_TAR_SPEED_MIN = 0.0
BXI_TAR_SPEED_MAX = 1.5
BXI_FALL_TERMINATION_HEIGHT = 0.2

# TienKung locomotion regularization, expressed as ProtoMotions reward terms.
# Keep these experiment-local: generic Steering intentionally stays
# robot-agnostic, while the ankle/stance terms below rely on ELF3 semantics.
BXI_SOFT_JOINT_LIMIT_FACTOR = 0.9
BXI_LIMIT_MAX_VIOLATION = 1.0
# TienKung's positive tracking terms total roughly 10, versus Steering's
# unit-scale task reward. Start its negative terms at one tenth strength.
BXI_JOINT_POS_LIMIT_WEIGHT = -0.2
BXI_PD_TARGET_LIMIT_WEIGHT = -0.5
BXI_ANKLE_ACTION_WEIGHT = -0.0001
BXI_ANKLE_TORQUE_WEIGHT = -0.00005
BXI_ACTION_RATE_WEIGHT = -0.001
BXI_FEET_Y_DISTANCE_WEIGHT = -0.2
BXI_ZERO_COMMAND_JOINT_DEVIATION_WEIGHT = -0.002
BXI_FEET_Y_TARGET_DISTANCE = 0.299
BXI_COMMAND_SPEED_THRESHOLD = 0.1
BXI_FACING_TOLERANCE = 0.2

# The data loaders do not need BXI-specific copies.
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config


def terrain_config(args: argparse.Namespace) -> TerrainConfig:
    """Build the base flat terrain with a 0.6 effective nominal friction.

    Training randomizes the robot material.  A unit-friction terrain with
    multiply combining therefore preserves both the nominal 0.6 coefficient
    and the sampled friction ranges without shifting them.
    """

    terrain_cfg = _base.terrain_config(args)
    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.MULTIPLY,
    )
    return terrain_cfg


def _steering_obs_component(*, use_noisy: bool):
    """Build Steering commands in the selected robot-orientation frame."""

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs import compute_steering_obs

    state = EnvContext.noisy if use_noisy else EnvContext.current
    return MdpComponent(
        compute_func=compute_steering_obs,
        dynamic_vars={
            "root_rot": state.root_rot,
            "tar_dir": EnvContext.steering.tar_dir,
            "tar_speed": EnvContext.steering.tar_speed,
            "tar_face_dir": EnvContext.steering.tar_face_dir,
        },
    )


def _soft_joint_limits(robot_cfg: RobotConfig):
    """Shrink hard limits about their midpoint to the TienKung 90% range."""

    soft_limit_factor = getattr(
        robot_cfg.control,
        "soft_pos_limit",
        BXI_SOFT_JOINT_LIMIT_FACTOR,
    )
    if not 0.0 < soft_limit_factor <= 1.0:
        raise ValueError(
            "elf3_bxi Steering soft joint limit factor must be in (0, 1], "
            f"got {soft_limit_factor}"
        )
    hard_lower = robot_cfg.kinematic_info.dof_limits_lower
    hard_upper = robot_cfg.kinematic_info.dof_limits_upper
    midpoint = 0.5 * (hard_lower + hard_upper)
    half_range = 0.5 * soft_limit_factor * (hard_upper - hard_lower)
    return midpoint - half_range, midpoint + half_range


def _ankle_joint_indices(robot_cfg: RobotConfig) -> torch.Tensor:
    indices = [
        index
        for index, name in enumerate(robot_cfg.kinematic_info.dof_names)
        if name.endswith(("_ankle_y_joint", "_ankle_x_joint"))
    ]
    if not indices:
        raise ValueError("elf3_bxi Steering could not resolve ankle joint indices")
    return torch.tensor(indices, dtype=torch.long)


def _single_semantic_body_index(robot_cfg: RobotConfig, semantic_name: str) -> int:
    body_names = robot_cfg.common_naming_to_robot_body_names[semantic_name]
    if len(body_names) != 1:
        raise ValueError(
            f"elf3_bxi Steering requires one body for {semantic_name!r}, "
            f"got {body_names!r}"
        )
    try:
        return robot_cfg.kinematic_info.body_names.index(body_names[0])
    except ValueError as exc:
        raise ValueError(
            f"elf3_bxi Steering body {body_names[0]!r} from {semantic_name!r} "
            "is absent from the robot asset"
        ) from exc


def _regularization_reward_components(robot_cfg: RobotConfig):
    """Build the safety and stance costs retained from TienKung locomotion."""

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards import (
        compute_action_l1,
        compute_action_rate_l2,
        compute_feet_y_distance_rew,
        compute_pow_rew,
        compute_soft_pos_limit_rew,
        compute_zero_command_joint_deviation_rew,
    )

    soft_lower, soft_upper = _soft_joint_limits(robot_cfg)
    ankle_indices = _ankle_joint_indices(robot_cfg)
    left_foot_index = _single_semantic_body_index(
        robot_cfg, "all_left_foot_bodies"
    )
    right_foot_index = _single_semantic_body_index(
        robot_cfg, "all_right_foot_bodies"
    )

    return {
        # Actual joint state must stay inside a 90%-of-hard-range envelope.
        "joint_pos_limits": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            static_params={
                "weight": BXI_JOINT_POS_LIMIT_WEIGHT,
                "dof_limits_lower": soft_lower,
                "dof_limits_upper": soft_upper,
                "max_violation": BXI_LIMIT_MAX_VIOLATION,
            },
        ),
        # Penalize the command itself as well: a hard stop can keep q legal
        # while an out-of-range PD target still requests saturated torque.
        "pd_target_limits": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current_processed_action,
            },
            static_params={
                "weight": BXI_PD_TARGET_LIMIT_WEIGHT,
                "dof_limits_lower": soft_lower,
                "dof_limits_upper": soft_upper,
                "max_violation": BXI_LIMIT_MAX_VIOLATION,
            },
        ),
        "ankle_action": MdpComponent(
            compute_func=compute_action_l1,
            dynamic_vars={"action": EnvContext.current_action},
            static_params={
                "weight": BXI_ANKLE_ACTION_WEIGHT,
                "joint_indices": ankle_indices,
                "min_value": -0.1,
                "zero_during_grace_period": True,
            },
        ),
        "ankle_torque": MdpComponent(
            compute_func=compute_pow_rew,
            dynamic_vars={
                "dof_forces": EnvContext.current.dof_forces,
                "dof_vel": EnvContext.current.dof_vel,
            },
            static_params={
                "weight": BXI_ANKLE_TORQUE_WEIGHT,
                "use_torque_squared": True,
                "joint_indices": ankle_indices,
                "min_value": -0.5,
                "zero_during_grace_period": True,
            },
        ),
        "action_rate": MdpComponent(
            compute_func=compute_action_rate_l2,
            dynamic_vars={
                "current_action": EnvContext.current_action,
                "previous_action": EnvContext.previous_action,
            },
            static_params={
                "weight": BXI_ACTION_RATE_WEIGHT,
                "min_value": -0.25,
                "zero_during_grace_period": True,
            },
        ),
        "feet_y_distance": MdpComponent(
            compute_func=compute_feet_y_distance_rew,
            dynamic_vars={
                "rigid_body_pos": EnvContext.current.rigid_body_pos,
                "anchor_rot": EnvContext.current.anchor_rot,
                "tar_dir": EnvContext.steering.tar_dir,
                "tar_speed": EnvContext.steering.tar_speed,
            },
            static_params={
                "weight": BXI_FEET_Y_DISTANCE_WEIGHT,
                "left_foot_body_index": left_foot_index,
                "right_foot_body_index": right_foot_index,
                "target_distance": BXI_FEET_Y_TARGET_DISTANCE,
                "lateral_speed_threshold": BXI_COMMAND_SPEED_THRESHOLD,
                "zero_during_grace_period": True,
            },
        ),
        "zero_command_joint_deviation": MdpComponent(
            compute_func=compute_zero_command_joint_deviation_rew,
            dynamic_vars={
                "dof_pos": EnvContext.current.dof_pos,
                "tar_speed": EnvContext.steering.tar_speed,
                "anchor_rot": EnvContext.current.anchor_rot,
                "tar_face_dir": EnvContext.steering.tar_face_dir,
            },
            static_params={
                "weight": BXI_ZERO_COMMAND_JOINT_DEVIATION_WEIGHT,
                "default_dof_pos": robot_cfg.default_dof_pos,
                "speed_threshold": BXI_COMMAND_SPEED_THRESHOLD,
                "facing_tolerance": BXI_FACING_TOLERANCE,
            },
        ),
    }


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Apply the BXI command curriculum, action mapping, and fall reset."""

    from protomotions.envs.action import make_bm_pd_action_config
    from protomotions.envs.component_factories import (
        fall_termination_factory,
        historical_reduced_coords_obs_factory,
        previous_actions_factory,
    )

    env_cfg = _base.env_config(robot_cfg, args)

    steering_cfg = env_cfg.control_components["steering"]
    steering_cfg.tar_speed_min = BXI_TAR_SPEED_MIN
    steering_cfg.tar_speed_max = BXI_TAR_SPEED_MAX
    # The first-stage references turn while facing the travel direction; they
    # do not contain independent-facing strafing examples.
    steering_cfg.enable_rand_facing = False

    # Keep the base clean max-coordinate observations for the privileged critic
    # and, critically, for both sides of the AMP discriminator.  Actor-only
    # inputs are quantities available on the deployed robot and consume the
    # observation-noise DR installed during configuration.  This also prevents
    # the discriminator from separating policy and expert data by sensor noise.
    env_cfg.observation_components["noisy_historical_reduced_coords_obs"] = (
        historical_reduced_coords_obs_factory(use_noisy=True)
    )
    env_cfg.observation_components["previous_actions"] = previous_actions_factory(
        history_steps=8
    )
    env_cfg.observation_components["noisy_steering"] = _steering_obs_component(
        use_noisy=True
    )

    env_cfg.action_config = make_bm_pd_action_config(robot_cfg)
    env_cfg.reward_components = {
        **env_cfg.reward_components,
        **_regularization_reward_components(robot_cfg),
    }
    env_cfg.termination_components = {
        **env_cfg.termination_components,
        "fall": fall_termination_factory(
            termination_height=BXI_FALL_TERMINATION_HEIGHT
        ),
    }
    return env_cfg


def agent_config(robot_cfg: RobotConfig, env_cfg: EnvConfig, args: argparse.Namespace):
    """Route noisy observations only to the actor, retaining a clean AMP path."""

    agent_cfg = _base.agent_config(robot_cfg, env_cfg, args)

    actor_keys = [
        "noisy_historical_reduced_coords_obs",
        "previous_actions",
        "noisy_steering",
    ]
    agent_cfg.model.actor.in_keys = actor_keys
    agent_cfg.model.actor.mu_model.in_keys = actor_keys

    # The top-level container must declare the union consumed by actor, critic,
    # discriminator, and discriminator critic.  The base clean keys remain for
    # all non-actor paths and match reference_obs_components exactly.
    agent_cfg.model.in_keys = [
        "noisy_historical_reduced_coords_obs",
        "previous_actions",
        "noisy_steering",
        "max_coords_obs",
        "steering",
        "historical_max_coords_obs",
    ]
    return agent_cfg


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
) -> None:
    """Apply the proven BXI robustness setup plus Steering fall sensing."""

    _validate_bxi_asset(robot_cfg)

    # The simulator factory shares this object with the robot's backend
    # simulation_params.  Override the live experiment layer without changing
    # Elf3BxiRobotConfig's 500/10 default used by BeyondMimic.
    simulator_cfg.sim.fps = BXI_PHYSICS_FPS
    simulator_cfg.sim.decimation = BXI_CONTROL_DECIMATION
    _validate_bxi_training_timing(simulator_cfg)

    # Reuse the public reset/contact/DR setup, avoiding the BeyondMimic BXI
    # overlay whose training contract deliberately requires 500/10.
    _robustness.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    simulator_cfg.default_robot_friction = BXI_NOMINAL_FRICTION

    left_feet = robot_cfg.common_naming_to_robot_body_names[
        "all_left_foot_bodies"
    ]
    right_feet = robot_cfg.common_naming_to_robot_body_names[
        "all_right_foot_bodies"
    ]
    # fall_termination needs contacts from every rigid body.  Only the feet are
    # exempt from triggering it.
    robot_cfg.update_fields(
        contact_bodies="all",
        non_termination_contact_bodies=[*left_feet, *right_feet],
    )


def _validate_bxi_asset(robot_cfg: RobotConfig) -> None:
    """Fail early when this experiment is paired with a non-BXI asset."""

    asset_file = getattr(getattr(robot_cfg, "asset", None), "asset_file_name", None)
    if asset_file != BXI_ASSET_FILE:
        raise ValueError(
            "mlp_bxi.py requires --robot-name elf3_bxi "
            f"(expected asset {BXI_ASSET_FILE!r}, got {asset_file!r})"
        )


def _validate_bxi_training_timing(simulator_cfg: SimulatorConfig) -> None:
    """Require the Steering-local physics rate while preserving a 20 ms action."""

    timing = (simulator_cfg.sim.fps, simulator_cfg.sim.decimation)
    expected = (BXI_PHYSICS_FPS, BXI_CONTROL_DECIMATION)
    if timing != expected:
        raise ValueError(
            "elf3_bxi Steering requires local training fps/decimation="
            f"{expected}, got {timing}"
        )


def _validate_bxi_inference_layer(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig
) -> None:
    """Validate a frozen BXI config while allowing backend timing conversion."""

    _validate_bxi_asset(robot_cfg)

    fps = simulator_cfg.sim.fps
    decimation = simulator_cfg.sim.decimation
    if fps <= 0 or decimation <= 0:
        raise ValueError(
            "elf3_bxi inference timing must be positive, got "
            f"fps/decimation={(fps, decimation)}"
        )
    control_dt = decimation / fps
    if not math.isclose(control_dt, BXI_CONTROL_DT, abs_tol=1e-9):
        raise ValueError(
            f"elf3_bxi inference requires control_dt={BXI_CONTROL_DT} seconds: "
            f"got fps/decimation={(fps, decimation)} "
            f"(control_dt={control_dt})"
        )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg,
    agent_cfg,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
) -> None:
    """Disable training-only randomness/termination and restore BXI material."""

    _validate_bxi_inference_layer(robot_cfg, simulator_cfg)
    _base.apply_inference_overrides(
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )

    if env_cfg is not None:
        from protomotions.envs.component_factories import (
            historical_reduced_coords_obs_factory,
        )

        env_cfg.termination_components = {}
        # Disabling observation DR makes StateHistoryBuffer stop storing a
        # separate noisy history.  Preserve the actor input names expected by
        # the checkpoint while rebinding them to clean runtime state.
        env_cfg.observation_components["noisy_historical_reduced_coords_obs"] = (
            historical_reduced_coords_obs_factory(use_noisy=False)
        )
        env_cfg.observation_components["noisy_steering"] = (
            _steering_obs_component(use_noisy=False)
        )

    simulator_cfg.domain_randomization = None
    simulator_cfg.default_robot_friction = BXI_NOMINAL_FRICTION

    # No observation or termination consumes contacts at inference, so avoid
    # constructing the all-body sensors baked into the training config.
    robot_cfg.contact_bodies = None

    if terrain_cfg is not None and terrain_cfg.sim_config is not None:
        terrain_cfg.sim_config.static_friction = BXI_NOMINAL_FRICTION
        terrain_cfg.sim_config.dynamic_friction = BXI_NOMINAL_FRICTION
        terrain_cfg.sim_config.restitution = 0.0
        terrain_cfg.sim_config.combine_mode = CombineMode.AVERAGE


__all__ = (
    "BXI_ASSET_FILE",
    "BXI_NOMINAL_FRICTION",
    "BXI_PHYSICS_FPS",
    "BXI_CONTROL_DECIMATION",
    "BXI_CONTROL_DT",
    "BXI_TAR_SPEED_MIN",
    "BXI_TAR_SPEED_MAX",
    "BXI_FALL_TERMINATION_HEIGHT",
    "BXI_SOFT_JOINT_LIMIT_FACTOR",
    "BXI_LIMIT_MAX_VIOLATION",
    "BXI_JOINT_POS_LIMIT_WEIGHT",
    "BXI_PD_TARGET_LIMIT_WEIGHT",
    "BXI_ANKLE_ACTION_WEIGHT",
    "BXI_ANKLE_TORQUE_WEIGHT",
    "BXI_ACTION_RATE_WEIGHT",
    "BXI_FEET_Y_DISTANCE_WEIGHT",
    "BXI_ZERO_COMMAND_JOINT_DEVIATION_WEIGHT",
    "BXI_FEET_Y_TARGET_DISTANCE",
    "BXI_COMMAND_SPEED_THRESHOLD",
    "BXI_FACING_TOLERANCE",
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "agent_config",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
)
