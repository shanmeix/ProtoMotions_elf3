# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export an ELF3 BXI ProtoMotions Steering policy to a unified ONNX graph.

The exported graph contains the observation kernels, the deterministic actor
(``mean_action``), and the BeyondMimic PD target conversion.  It is intentionally
simulator-free: the resolved inference configuration and checkpoint are enough.

The deployment contract for the ELF3 BXI stage-1 Steering policy is fixed to:

* eight previous 20 ms robot-state samples (newest first),
* eight previous raw policy actions (newest first),
* a world-frame steering direction, speed, and facing direction,
* 200 Hz physics with decimation 4 (50 Hz policy), and
* 29 raw actions / joint-position targets.

Typical usage::

    python deployment/export_steering_onnx.py \
        --checkpoint results/elf3_bxi_steering_stage1_200hz/epoch_30000.ckpt \
        --output results/elf3_bxi_steering_stage1_200hz/compiled_models/epoch_30000

This writes ``unified_pipeline.onnx`` and ``unified_pipeline.yaml``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


EXPECTED_ACTOR_OBS_KEYS = (
    "noisy_historical_reduced_coords_obs",
    "previous_actions",
    "noisy_steering",
)

# ``historical.root_rot`` is present in the MDP component binding, but the
# reduced-coordinates kernel does not consume it unless root-velocity features
# are enabled.  The legacy ONNX exporter therefore removes it as dead input.
EXPECTED_OBS_CONTEXT_KEYS = (
    "current.root_rot",
    "historical.actions",
    "historical.anchor_rot",
    "historical.dof_pos",
    "historical.dof_vel",
    "historical.root_local_ang_vel",
    "historical.root_rot",
    "steering.tar_dir",
    "steering.tar_face_dir",
    "steering.tar_speed",
)
EXPECTED_ONNX_CONTEXT_KEYS = tuple(
    key for key in EXPECTED_OBS_CONTEXT_KEYS if key != "historical.root_rot"
)
EXPECTED_ONNX_OUTPUTS = (
    "actions",
    "joint_pos_targets",
    "stiffness_targets",
    "damping_targets",
)

ELF3_NUM_DOFS = 29
STEERING_HISTORY_STEPS = 8
STEERING_PHYSICS_FPS = 200
STEERING_DECIMATION = 4


# ---------------------------------------------------------------------------
# Simulator-free sample context
# ---------------------------------------------------------------------------


def _random_quaternion(*shape: int):
    import torch
    import torch.nn.functional as F

    return F.normalize(torch.randn(*shape), dim=-1)


class _MockCurrent:
    def __init__(self, num_envs: int, anchor_idx: int):
        self.anchor_idx = anchor_idx
        self.root_rot = _random_quaternion(num_envs, 4)


class _MockHistorical:
    def __init__(self, num_envs: int, history_steps: int, num_dofs: int):
        import torch

        # These tensors represent the HistoricalView, not the backing buffer.
        # Index 0 is t-dt and index H-1 is t-H*dt; current state t is excluded.
        self.dof_pos = torch.randn(num_envs, history_steps, num_dofs)
        self.dof_vel = torch.randn(num_envs, history_steps, num_dofs)
        self.root_rot = _random_quaternion(num_envs, history_steps, 4)
        self.root_local_ang_vel = torch.randn(num_envs, history_steps, 3)
        self.anchor_rot = _random_quaternion(num_envs, history_steps, 4)
        self.actions = torch.randn(num_envs, history_steps, num_dofs)


class _MockSteering:
    def __init__(self, num_envs: int):
        import torch
        import torch.nn.functional as F

        # Keep direction and facing as independent tensors.  Reusing one tensor
        # can alias graph inputs during tracing even though stage 1 commands set
        # both values equal at runtime.
        self.tar_dir = F.normalize(torch.randn(num_envs, 2), dim=-1)
        self.tar_face_dir = F.normalize(torch.randn(num_envs, 2), dim=-1)
        self.tar_speed = torch.rand(num_envs)


class MockContext:
    """Minimal EnvContext shape contract used only for tracing and tests."""

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        history_steps: int,
        anchor_idx: int = 0,
    ):
        self.current = _MockCurrent(num_envs, anchor_idx)
        self.historical = _MockHistorical(num_envs, history_steps, num_dofs)
        self.steering = _MockSteering(num_envs)


# ---------------------------------------------------------------------------
# Small reusable helpers
# ---------------------------------------------------------------------------


def _canonicalize_actor_observations(
    actor_observation_configs,
    *,
    all_observation_configs,
    obs_pairs=None,
):
    """Reuse BM's frozen-config-safe noisy-to-measured-state conversion."""

    try:
        from deployment.export_bm_tracker_onnx import (
            _canonicalize_deployment_observation_configs,
        )
    except ImportError:  # Direct ``python deployment/export_steering_onnx.py``.
        from export_bm_tracker_onnx import (  # type: ignore[no-redef]
            _canonicalize_deployment_observation_configs,
        )

    return _canonicalize_deployment_observation_configs(
        actor_observation_configs,
        all_observation_configs=all_observation_configs,
        obs_pairs=obs_pairs,
    )


def _resolve_attr_path(path: str, obj):
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _sanitize(name: str) -> str:
    return (
        name.replace(".", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(":", "_")
    )


def _strip_numeric_suffix(name: str) -> str:
    """Remove any number of ONNX ``.N``/``_N`` collision suffixes."""

    result = name
    while True:
        for separator in (".", "_"):
            prefix, found, suffix = result.rpartition(separator)
            if found and prefix and suffix.isdigit():
                result = prefix
                break
        else:
            return result


def _map_actual_onnx_inputs(actual_names: list[str], semantic_keys: list[str]) -> dict:
    sanitized_to_key = {_sanitize(key): key for key in semantic_keys}
    mapping = {}
    for actual_name in actual_names:
        candidate = actual_name
        if candidate not in sanitized_to_key:
            candidate = _strip_numeric_suffix(candidate)
        if candidate not in sanitized_to_key:
            raise ValueError(
                f"Cannot map ONNX input {actual_name!r} to one of {semantic_keys}"
            )
        mapping[actual_name] = sanitized_to_key[candidate]
    return mapping


def _get_field(config: Any, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_float_list(value, *, name: str, expected_size: int) -> list[float]:
    if value is None:
        raise ValueError(f"action_config is missing required field {name!r}")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        result = [float(item) for item in value.reshape(-1).tolist()]
    elif isinstance(value, (list, tuple)):
        result = [float(item) for item in value]
    else:
        result = [float(value)]
    if len(result) != expected_size:
        raise ValueError(
            f"{name} has {len(result)} values; expected {expected_size} ELF3 joints"
        )
    return result


def _load_resolved_configs(checkpoint_path: Path):
    import torch

    inference_path = checkpoint_path.parent / "resolved_configs_inference.pt"
    resolved_path = inference_path
    if not resolved_path.exists():
        resolved_path = checkpoint_path.parent / "resolved_configs.pt"
        log.warning(
            "%s is missing; falling back to %s and canonicalizing noisy actor "
            "bindings in memory",
            inference_path,
            resolved_path,
        )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find resolved_configs_inference.pt or resolved_configs.pt "
            f"beside {checkpoint_path}"
        )
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)
    return resolved, resolved_path


def _frozen_timing(simulator_config) -> dict[str, float | int]:
    sim = getattr(simulator_config, "sim", None)
    fps = int(getattr(sim, "fps", 0) or 0)
    decimation = int(getattr(sim, "decimation", 0) or 0)
    if (fps, decimation) != (STEERING_PHYSICS_FPS, STEERING_DECIMATION):
        raise ValueError(
            "ELF3 BXI Steering deployment is frozen to IsaacLab 200 Hz physics "
            f"with decimation 4, but the resolved checkpoint records {fps}/{decimation}"
        )
    physics_dt = 1.0 / fps
    control_dt = decimation / fps
    return {
        "physics_fps": fps,
        "physics_dt": physics_dt,
        "decimation": decimation,
        "control_dt": control_dt,
        "policy_hz": 1.0 / control_dt,
    }


def _reconstruct_actor(actor_config, checkpoint_payload, mock_obs_td):
    """Materialize a lazy actor and strictly load its ``_actor.`` checkpoint."""

    import torch
    from protomotions.utils.hydra_replacement import get_class

    ActorClass = get_class(actor_config._target_)
    actor = ActorClass(actor_config)
    actor.eval()
    with torch.no_grad():
        actor(mock_obs_td.clone())

    model_state = checkpoint_payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint does not contain a model state dictionary")
    actor_state = {
        key.removeprefix("_actor."): value
        for key, value in model_state.items()
        if key.startswith("_actor.")
    }
    if not actor_state:
        raise ValueError("checkpoint model contains no '_actor.' weights")
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    return actor


def _validate_actor_contract(actor_in_keys, obs_shapes: dict[str, list[int]]) -> None:
    if tuple(actor_in_keys) != EXPECTED_ACTOR_OBS_KEYS:
        raise ValueError(
            "This exporter only supports the ELF3 BXI Steering actor inputs "
            f"{list(EXPECTED_ACTOR_OBS_KEYS)}, got {list(actor_in_keys)}"
        )
    expected_shapes = {
        "noisy_historical_reduced_coords_obs": [1, 512],
        "previous_actions": [1, 232],
        "noisy_steering": [1, 5],
    }
    if obs_shapes != expected_shapes:
        raise ValueError(
            f"ELF3 BXI Steering observation shapes changed: {obs_shapes}; "
            f"expected {expected_shapes}"
        )


# ---------------------------------------------------------------------------
# Validation and sidecar
# ---------------------------------------------------------------------------


def _run_ort_validation(
    *,
    session,
    unified,
    sample_inputs,
    obs_input_keys,
    onnx_name_to_key,
    actual_output_names,
    pd_action_offset,
    action_scale,
    stiffness,
    damping,
    tolerance: float = 1e-4,
) -> dict[str, float]:
    """Compare PyTorch/ORT for batches 1 and 2 and verify the PD contract."""

    import numpy as np
    import torch

    input_by_key = dict(zip(obs_input_keys, sample_inputs))
    max_diffs: dict[str, float] = {}

    for batch_size in (1, 2):
        if batch_size == 1:
            torch_inputs = sample_inputs
        else:
            torch_inputs = [torch.cat((value, value.clone()), dim=0) for value in sample_inputs]

        batch_by_key = dict(zip(obs_input_keys, torch_inputs))
        ort_inputs = {
            onnx_name: batch_by_key[semantic_key].detach().cpu().numpy()
            for onnx_name, semantic_key in onnx_name_to_key.items()
        }
        with torch.no_grad():
            pytorch_outputs = unified(*torch_inputs)
        ort_outputs = session.run(actual_output_names, ort_inputs)

        if len(ort_outputs) != len(EXPECTED_ONNX_OUTPUTS):
            raise ValueError(
                f"ONNX returned {len(ort_outputs)} outputs; expected 4"
            )
        for name, ort_value, torch_value in zip(
            EXPECTED_ONNX_OUTPUTS, ort_outputs, pytorch_outputs
        ):
            if ort_value.shape[0] != batch_size:
                raise ValueError(
                    f"Dynamic batch failed for {name}: expected {batch_size}, "
                    f"got shape {ort_value.shape}"
                )
            diff = float(
                np.max(np.abs(ort_value - torch_value.detach().cpu().numpy()))
            )
            max_diffs[f"batch_{batch_size}.{name}"] = diff
            if not np.isfinite(diff) or diff > tolerance:
                raise ValueError(
                    f"PyTorch/ONNX mismatch for {name}, batch {batch_size}: "
                    f"max_diff={diff:.3e} > {tolerance:.3e}"
                )

        offset = np.asarray(pd_action_offset, dtype=np.float32).reshape(1, -1)
        scale = np.asarray(action_scale, dtype=np.float32).reshape(1, -1)
        expected_targets = offset + scale * ort_outputs[0]
        target_diff = float(np.max(np.abs(expected_targets - ort_outputs[1])))
        max_diffs[f"batch_{batch_size}.pd_reconstruction"] = target_diff
        if not np.isfinite(target_diff) or target_diff > tolerance:
            raise ValueError(
                "Sidecar pd_action_offset/action_scale do not reproduce ONNX "
                f"joint_pos_targets (max_diff={target_diff:.3e})"
            )

        expected_stiffness = np.broadcast_to(
            np.asarray(stiffness, dtype=np.float32).reshape(1, -1),
            ort_outputs[2].shape,
        )
        expected_damping = np.broadcast_to(
            np.asarray(damping, dtype=np.float32).reshape(1, -1),
            ort_outputs[3].shape,
        )
        for name, expected, actual in (
            ("stiffness_targets", expected_stiffness, ort_outputs[2]),
            ("damping_targets", expected_damping, ort_outputs[3]),
        ):
            gain_diff = float(np.max(np.abs(expected - actual)))
            max_diffs[f"batch_{batch_size}.{name}_sidecar"] = gain_diff
            if not np.isfinite(gain_diff) or gain_diff > tolerance:
                raise ValueError(
                    f"Sidecar {name} values differ from ONNX (max_diff={gain_diff:.3e})"
                )

    # Keep this explicit so a future refactor cannot accidentally validate with
    # values that were not the sample values used for the exported graph.
    assert set(input_by_key) == set(obs_input_keys)
    return max_diffs


def _input_descriptor(
    *,
    onnx_name: str,
    key: str,
    shape: list[int],
    joint_names: list[str],
    anchor_body_name: str,
    history_steps: int,
    history_offsets_seconds: list[float],
    steering_speed_min: float,
    steering_speed_max: float,
) -> dict:
    entry: dict[str, Any] = {"name": onnx_name, "key": key, "shape": shape}
    history_common = {
        "history": history_steps,
        "include_current_value_in_history": False,
        "history_order": "newest_to_oldest",
        "time_offsets_seconds": history_offsets_seconds,
    }

    if key == "current.root_rot":
        entry.update(
            kind="root_body_rot",
            frame="world",
            body_name=anchor_body_name,
            quaternion_order="xyzw",
            element_names=[["x", "y", "z", "w"]],
        )
    elif key == "historical.actions":
        entry.update(
            kind="last_actions",
            representation="raw_policy_action",
            output_key="actions",
            element_names=[joint_names],
            **history_common,
        )
    elif key == "historical.anchor_rot":
        entry.update(
            kind="anchor_rot",
            frame="world",
            body_name=anchor_body_name,
            quaternion_order="xyzw",
            element_names=[["x", "y", "z", "w"]],
            **history_common,
        )
    elif key == "historical.dof_pos":
        entry.update(kind="joint_pos", element_names=[joint_names], **history_common)
    elif key == "historical.dof_vel":
        entry.update(kind="joint_vel", element_names=[joint_names], **history_common)
    elif key == "historical.root_local_ang_vel":
        entry.update(
            kind="local_root_ang_vel",
            frame="root_local",
            units="rad/s",
            element_names=[["x", "y", "z"]],
            **history_common,
        )
    elif key == "steering.tar_dir":
        entry.update(
            kind="steering_target_direction",
            frame="world",
            normalization="unit_xy",
            element_names=[["x", "y"]],
        )
    elif key == "steering.tar_face_dir":
        entry.update(
            kind="steering_target_facing_direction",
            frame="world",
            normalization="unit_xy",
            element_names=[["x", "y"]],
            note="Stage 1 was trained with facing direction equal to movement direction.",
        )
    elif key == "steering.tar_speed":
        entry.update(
            kind="steering_target_speed",
            units="m/s",
            minimum=steering_speed_min,
            maximum=steering_speed_max,
        )
    else:
        raise ValueError(f"No Steering sidecar descriptor for ONNX input {key!r}")
    return entry


def _build_yaml(
    *,
    actual_input_names: list[str],
    actual_output_names: list[str],
    onnx_name_to_key: dict[str, str],
    input_shapes: dict[str, list[int]],
    output_shapes: dict[str, list[int]],
    obs_context_keys: list[str],
    actor_obs_shapes: dict[str, list[int]],
    joint_names: list[str],
    body_names: list[str],
    anchor_body_name: str,
    anchor_body_index: int,
    mjcf_path: str,
    timing: dict[str, float | int],
    history_steps: int,
    pd_action_offset: list[float],
    action_scale: list[float],
    stiffness: list[float],
    damping: list[float],
    effort_limits: list[float] | None,
    pd_target_max_accel: float | None,
    steering_speed_min: float,
    steering_speed_max: float,
    enable_rand_facing: bool,
    checkpoint: str,
    resolved_config: str,
    validation_max_diffs: dict[str, float] | None,
) -> dict:
    control_dt = float(timing["control_dt"])
    history_offsets_seconds = [
        round((step + 1) * control_dt, 6) for step in range(history_steps)
    ]

    policy_inputs = [
        _input_descriptor(
            onnx_name=name,
            key=onnx_name_to_key[name],
            shape=input_shapes[onnx_name_to_key[name]],
            joint_names=joint_names,
            anchor_body_name=anchor_body_name,
            history_steps=history_steps,
            history_offsets_seconds=history_offsets_seconds,
            steering_speed_min=steering_speed_min,
            steering_speed_max=steering_speed_max,
        )
        for name in actual_input_names
    ]

    output_kinds = {
        "actions": "actions",
        "joint_pos_targets": "joint_pos_targets",
        "stiffness_targets": "stiffness_targets",
        "damping_targets": "damping_targets",
    }
    policy_outputs = []
    for name in actual_output_names:
        semantic_name = _strip_numeric_suffix(name)
        if semantic_name not in output_kinds:
            raise ValueError(f"No Steering sidecar descriptor for output {name!r}")
        entry: dict[str, Any] = {
            "name": name,
            "key": name,
            "kind": output_kinds[semantic_name],
            "shape": output_shapes[semantic_name],
        }
        if semantic_name != "actions":
            entry["joint_names"] = joint_names
        policy_outputs.append(entry)

    elided_context_keys = [
        key for key in obs_context_keys if key not in onnx_name_to_key.values()
    ]

    return {
        "type": "unified_pipeline",
        "task": "steering",
        "dt": control_dt,
        "joint_names": joint_names,
        "body_names": body_names,
        "default_joint_stiffness": stiffness,
        "default_joint_damping": damping,
        "policy_inputs": policy_inputs,
        "policy_outputs": policy_outputs,
        "_runtime": {
            "onnx_in_names": actual_input_names,
            "onnx_out_names": actual_output_names,
            "onnx_name_to_in_key": onnx_name_to_key,
            "obs_context_keys": obs_context_keys,
            "elided_context_keys": elided_context_keys,
        },
        "metadata": {
            "checkpoint": checkpoint,
            "resolved_config": resolved_config,
            "control_type": "bm_pd_action",
            "actor_observation_keys": list(EXPECTED_ACTOR_OBS_KEYS),
            "actor_observation_shapes": actor_obs_shapes,
            "actor_input_dim": sum(shape[-1] for shape in actor_obs_shapes.values()),
            "validation_max_diffs": validation_max_diffs,
        },
        "robot": {
            "mjcf_path": mjcf_path,
            "num_bodies": len(body_names),
            "num_dofs": len(joint_names),
            "anchor_body_name": anchor_body_name,
            "anchor_body_index": anchor_body_index,
            "root_body_name": body_names[0],
            "root_body_index": 0,
            "body_names": body_names,
            "joint_names": joint_names,
        },
        "control": {
            "stiffness": stiffness,
            "damping": damping,
            "effort_limits": effort_limits,
            "action_scale": action_scale,
            "pd_action_offset": pd_action_offset,
            "pd_target_max_accel": pd_target_max_accel,
            "raw_action_transform": "none",
            "joint_target_formula": "pd_action_offset + action_scale * actions",
            "action_ema_alpha": 1.0,
        },
        "timing": {
            **timing,
            "history_sample_dt": control_dt,
            "history_offsets_seconds": history_offsets_seconds,
        },
        "observation": {
            "history_steps": history_steps,
            "history_includes_current": False,
            "history_order": "newest_to_oldest",
            "state_per_step_layout": [
                {"name": "dof_pos", "size": len(joint_names)},
                {"name": "dof_vel", "size": len(joint_names)},
                {"name": "root_local_ang_vel", "size": 3},
                {"name": "projected_gravity", "size": 3},
            ],
            "state_per_step_dim": 2 * len(joint_names) + 6,
            "state_history_dim": history_steps * (2 * len(joint_names) + 6),
            "raw_action_history_dim": history_steps * len(joint_names),
            "steering_dim": 5,
        },
        "steering": {
            "target_speed_min": steering_speed_min,
            "target_speed_max": steering_speed_max,
            "direction_frame": "world",
            "facing_direction_frame": "world",
            "enable_random_facing_during_training": enable_rand_facing,
            "deployment_rule": (
                "Set tar_face_dir equal to tar_dir for this stage-1 checkpoint."
                if not enable_rand_facing
                else "Facing direction may be commanded independently."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main export flow
# ---------------------------------------------------------------------------


def export_steering(
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
) -> Path:
    import torch
    from tensordict import TensorDict

    from protomotions.utils.export_utils import (
        ActionExportModule,
        ObservationExportModule,
        UnifiedPipelineModule,
    )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    resolved, resolved_path = _load_resolved_configs(checkpoint_path)
    robot_config = resolved["robot"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]
    simulator_config = resolved["simulator"]

    num_dofs = int(robot_config.kinematic_info.num_dofs)
    if num_dofs != ELF3_NUM_DOFS:
        raise ValueError(f"ELF3 BXI Steering requires 29 DOFs, got {num_dofs}")
    history_steps = int(env_config.num_state_history_steps)
    if history_steps != STEERING_HISTORY_STEPS:
        raise ValueError(
            f"ELF3 BXI Steering requires 8 historical steps, got {history_steps}"
        )
    timing = _frozen_timing(simulator_config)

    actor_in_keys = list(agent_config.model.actor.in_keys)
    if tuple(actor_in_keys) != EXPECTED_ACTOR_OBS_KEYS:
        raise ValueError(
            f"Unexpected actor inputs {actor_in_keys}; expected "
            f"{list(EXPECTED_ACTOR_OBS_KEYS)}"
        )
    missing = set(actor_in_keys) - set(env_config.observation_components)
    if missing:
        raise ValueError(f"Actor observation components are missing: {sorted(missing)}")

    l2c2 = getattr(agent_config, "l2c2", None)
    actor_observation_configs = _canonicalize_actor_observations(
        {key: env_config.observation_components[key] for key in actor_in_keys},
        all_observation_configs=env_config.observation_components,
        obs_pairs=getattr(l2c2, "obs_pairs", None),
    )

    mock = MockContext(
        num_envs=1,
        num_dofs=num_dofs,
        history_steps=history_steps,
        anchor_idx=int(robot_config.anchor_body_index),
    )
    obs_module = ObservationExportModule(
        actor_observation_configs, mock, device="cpu"
    ).eval()
    obs_input_keys = obs_module.get_input_keys()
    if tuple(obs_input_keys) != EXPECTED_OBS_CONTEXT_KEYS:
        raise ValueError(
            f"Steering observation context changed: {obs_input_keys}; expected "
            f"{list(EXPECTED_OBS_CONTEXT_KEYS)}"
        )
    sample_inputs = [_resolve_attr_path(key, mock) for key in obs_input_keys]
    with torch.no_grad():
        obs_outputs = obs_module(*sample_inputs)
    obs_output_keys = obs_module.get_output_keys()
    actor_obs_shapes = {
        key: list(value.shape) for key, value in zip(obs_output_keys, obs_outputs)
    }
    _validate_actor_contract(actor_in_keys, actor_obs_shapes)

    mock_obs_td = TensorDict(
        dict(zip(obs_output_keys, obs_outputs)), batch_size=[1]
    )
    checkpoint_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    actor = _reconstruct_actor(
        agent_config.model.actor, checkpoint_payload, mock_obs_td
    )
    if list(actor.in_keys) != actor_in_keys:
        raise ValueError(
            f"Reconstructed actor in_keys {list(actor.in_keys)} do not match config "
            f"{actor_in_keys}"
        )

    action_config = env_config.action_config
    pd_action_offset = _as_float_list(
        _get_field(action_config, "pd_action_offset"),
        name="pd_action_offset",
        expected_size=num_dofs,
    )
    action_scale = _as_float_list(
        _get_field(action_config, "action_scale"),
        name="action_scale",
        expected_size=num_dofs,
    )
    stiffness = _as_float_list(
        _get_field(action_config, "stiffness"),
        name="stiffness",
        expected_size=num_dofs,
    )
    damping = _as_float_list(
        _get_field(action_config, "damping"),
        name="damping",
        expected_size=num_dofs,
    )
    action_fn = _get_field(action_config, "fn")
    if getattr(action_fn, "__name__", None) != "bm_pd_action":
        raise ValueError(
            "ELF3 BXI Steering export requires bm_pd_action, got "
            f"{getattr(action_fn, '__name__', action_fn)!r}"
        )

    action_module = ActionExportModule(action_config, device="cpu").eval()
    unified = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=actor,
        action_module=action_module,
        policy_in_keys=actor_in_keys,
        policy_action_key="mean_action",
    ).cpu().eval()

    with torch.no_grad():
        pytorch_outputs = unified(*sample_inputs)
    if [list(value.shape) for value in pytorch_outputs] != [[1, num_dofs]] * 4:
        raise ValueError(
            "Unexpected unified pipeline output shapes: "
            f"{[list(value.shape) for value in pytorch_outputs]}"
        )

    requested_input_names = [_sanitize(key) for key in obs_input_keys]
    onnx_path = output_path / "unified_pipeline.onnx"
    log.info("Exporting %s", onnx_path)
    torch.onnx.export(
        unified,
        tuple(sample_inputs),
        str(onnx_path),
        input_names=requested_input_names,
        output_names=list(EXPECTED_ONNX_OUTPUTS),
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in requested_input_names},
            **{name: {0: "batch_size"} for name in EXPECTED_ONNX_OUTPUTS},
        },
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    actual_input_names = [item.name for item in session.get_inputs()]
    actual_output_names = [item.name for item in session.get_outputs()]
    onnx_name_to_key = _map_actual_onnx_inputs(
        actual_input_names, obs_input_keys
    )
    if tuple(onnx_name_to_key.values()) != EXPECTED_ONNX_CONTEXT_KEYS:
        raise ValueError(
            "Exported Steering graph input contract changed: "
            f"{list(onnx_name_to_key.values())}; expected "
            f"{list(EXPECTED_ONNX_CONTEXT_KEYS)}"
        )
    if tuple(_strip_numeric_suffix(name) for name in actual_output_names) != (
        EXPECTED_ONNX_OUTPUTS
    ):
        raise ValueError(
            f"Exported graph outputs {actual_output_names}; expected "
            f"{list(EXPECTED_ONNX_OUTPUTS)}"
        )

    validation_max_diffs = None
    if validate:
        validation_max_diffs = _run_ort_validation(
            session=session,
            unified=unified,
            sample_inputs=sample_inputs,
            obs_input_keys=obs_input_keys,
            onnx_name_to_key=onnx_name_to_key,
            actual_output_names=actual_output_names,
            pd_action_offset=pd_action_offset,
            action_scale=action_scale,
            stiffness=stiffness,
            damping=damping,
        )
        log.info(
            "ONNX Runtime validation passed (max diff %.3e)",
            max(validation_max_diffs.values(), default=0.0),
        )

    joint_names = list(robot_config.kinematic_info.dof_names)
    body_names = list(robot_config.kinematic_info.body_names)
    effort_limits = None
    try:
        effort_limits = [
            float(robot_config.control.control_info[name].effort_limit)
            for name in joint_names
        ]
    except (AttributeError, KeyError, TypeError):
        pass

    steering_config = env_config.control_components.get("steering")
    if steering_config is None:
        raise ValueError("env.control_components does not contain 'steering'")
    speed_min = float(steering_config.tar_speed_min)
    speed_max = float(steering_config.tar_speed_max)
    enable_rand_facing = bool(steering_config.enable_rand_facing)

    input_shapes = {
        key: list(value.shape) for key, value in zip(obs_input_keys, sample_inputs)
    }
    output_shapes = {
        name: list(value.shape)
        for name, value in zip(EXPECTED_ONNX_OUTPUTS, pytorch_outputs)
    }
    yaml_content = _build_yaml(
        actual_input_names=actual_input_names,
        actual_output_names=actual_output_names,
        onnx_name_to_key=onnx_name_to_key,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        obs_context_keys=obs_input_keys,
        actor_obs_shapes=actor_obs_shapes,
        joint_names=joint_names,
        body_names=body_names,
        anchor_body_name=str(robot_config.anchor_body_name),
        anchor_body_index=int(robot_config.anchor_body_index),
        mjcf_path=str(robot_config.asset.asset_file_name),
        timing=timing,
        history_steps=history_steps,
        pd_action_offset=pd_action_offset,
        action_scale=action_scale,
        stiffness=stiffness,
        damping=damping,
        effort_limits=effort_limits,
        pd_target_max_accel=(
            None
            if getattr(simulator_config, "pd_target_max_accel", None) is None
            else float(simulator_config.pd_target_max_accel)
        ),
        steering_speed_min=speed_min,
        steering_speed_max=speed_max,
        enable_rand_facing=enable_rand_facing,
        checkpoint=str(checkpoint_path),
        resolved_config=str(resolved_path.resolve()),
        validation_max_diffs=validation_max_diffs,
    )

    import yaml

    yaml_path = output_path / "unified_pipeline.yaml"
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(yaml_content, stream, sort_keys=False)
    log.info("Wrote %s", yaml_path)
    return onnx_path


# Backwards-friendly spelling shared by the other deployment exporters.
export_tracker = export_steering


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export an ELF3 BXI Steering checkpoint to unified ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a *.ckpt file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <checkpoint_dir>/compiled_models/<checkpoint_stem>)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip PyTorch/ONNX Runtime numerical and dynamic-batch validation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output = args.output
    if output is None:
        checkpoint_path = Path(args.checkpoint)
        output = str(
            checkpoint_path.parent / "compiled_models" / checkpoint_path.stem
        )
    exported = export_steering(
        checkpoint=args.checkpoint,
        output_dir=output,
        validate=not args.no_validate,
    )
    log.info("Done: %s", exported)
    log.info("Sidecar: %s", exported.with_suffix(".yaml"))
