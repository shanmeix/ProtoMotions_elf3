# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BXI deployment-aligned overlay for the BeyondMimic L2C2 experiment.

Configuration ownership is intentionally split across two reusable layers:

* ``mlp_bm_l2c2`` owns the task, observations, rewards, agent, reset noise, and
  domain-randomization distributions.
* ``Elf3BxiRobotConfig`` owns the BXI asset, actuator configuration, and the
  500 Hz physics / 10-step decimation used for training. Inference backend
  conversion may choose a different physics rate while retaining the 20 ms
  policy period.

This overlay owns only the nominal BXI contact friction.  Keeping the timing in
the robot config is important because simulator switching reads
``robot_cfg.simulation_params`` directly.
"""

import argparse
import math

from examples.experiments.mimic import mlp_bm_l2c2 as _base
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import CombineMode, TerrainConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig


BXI_ASSET_FILE = "mjcf/elf3_bxi.xml"
BXI_NOMINAL_FRICTION = 0.6
BXI_PHYSICS_FPS = 500
BXI_CONTROL_DECIMATION = 10
BXI_CONTROL_DT = BXI_CONTROL_DECIMATION / BXI_PHYSICS_FPS

# The BeyondMimic task and learner stay single-sourced in mlp_bm_l2c2.py.
terrain_config = _base.terrain_config
scene_lib_config = _base.scene_lib_config
motion_lib_config = _base.motion_lib_config
env_config = _base.env_config
agent_config = _base.agent_config


def _validate_bxi_robot_layer(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    *,
    require_training_timing: bool,
) -> None:
    """Fail early when this overlay is paired with the wrong robot layer."""
    asset_file = getattr(getattr(robot_cfg, "asset", None), "asset_file_name", None)
    if asset_file != BXI_ASSET_FILE:
        raise ValueError(
            "mlp_bm_l2c2_bxi.py requires --robot-name elf3_bxi "
            f"(expected asset {BXI_ASSET_FILE!r}, got {asset_file!r})"
        )

    timing = (simulator_cfg.sim.fps, simulator_cfg.sim.decimation)
    if require_training_timing:
        expected_timing = (BXI_PHYSICS_FPS, BXI_CONTROL_DECIMATION)
        if timing != expected_timing:
            raise ValueError(
                "elf3_bxi must provide simulator timing through its robot config: "
                f"expected fps/decimation={expected_timing}, got {timing}"
            )
    else:
        fps, decimation = timing
        if fps <= 0 or decimation <= 0:
            raise ValueError(
                f"elf3_bxi inference timing must be positive, got {timing}"
            )
        control_dt = decimation / fps
        if not math.isclose(control_dt, BXI_CONTROL_DT, abs_tol=1e-9):
            raise ValueError(
                f"elf3_bxi inference requires control_dt={BXI_CONTROL_DT} seconds: "
                f"got fps/decimation={timing} (control_dt={control_dt})"
            )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    args: argparse.Namespace,
) -> None:
    """Apply the base robustness setup and BXI's nominal contact friction."""
    _validate_bxi_robot_layer(
        robot_cfg, simulator_cfg, require_training_timing=True
    )
    _base.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)

    # Training friction randomization comes from the base experiment.  This is
    # the non-randomized baseline used when DR is disabled and at inference.
    simulator_cfg.default_robot_friction = BXI_NOMINAL_FRICTION


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
    """Reuse base inference behavior, then restore the BXI nominal material."""
    _validate_bxi_robot_layer(
        robot_cfg, simulator_cfg, require_training_timing=False
    )
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

    simulator_cfg.default_robot_friction = BXI_NOMINAL_FRICTION
    if terrain_cfg is not None and terrain_cfg.sim_config is not None:
        # PhysX combines the two 0.6 materials to the same 0.6 effective
        # coefficient.  The BXI MuJoCo foot collision geoms also use 0.6.
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
    "terrain_config",
    "scene_lib_config",
    "motion_lib_config",
    "env_config",
    "agent_config",
    "configure_robot_and_simulator",
    "apply_inference_overrides",
)
