# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression tests for the ELF3 BXI Steering exporter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import deployment.export_steering_onnx as steering_export
from protomotions.envs.component_factories import (
    historical_reduced_coords_obs_factory,
    previous_actions_factory,
)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.steering import compute_steering_obs
from protomotions.utils.export_utils import ObservationExportModule


def _actor_observation_configs():
    return {
        "noisy_historical_reduced_coords_obs": (
            historical_reduced_coords_obs_factory(use_noisy=True)
        ),
        "previous_actions": previous_actions_factory(
            history_steps=steering_export.STEERING_HISTORY_STEPS,
            processed=False,
        ),
        "noisy_steering": MdpComponent(
            compute_func=compute_steering_obs,
            dynamic_vars={
                "root_rot": EnvContext.noisy.root_rot,
                "tar_dir": EnvContext.steering.tar_dir,
                "tar_speed": EnvContext.steering.tar_speed,
                "tar_face_dir": EnvContext.steering.tar_face_dir,
            },
        ),
    }


def _input_shapes():
    return {
        "current.root_rot": [1, 4],
        "historical.actions": [1, 8, 29],
        "historical.anchor_rot": [1, 8, 4],
        "historical.dof_pos": [1, 8, 29],
        "historical.dof_vel": [1, 8, 29],
        "historical.root_local_ang_vel": [1, 8, 3],
        "historical.root_rot": [1, 8, 4],
        "steering.tar_dir": [1, 2],
        "steering.tar_face_dir": [1, 2],
        "steering.tar_speed": [1],
    }


def test_mock_context_builds_exact_749_dim_actor_observation_contract():
    context = steering_export.MockContext(
        num_envs=2,
        num_dofs=29,
        history_steps=8,
        anchor_idx=0,
    )
    training_configs = _actor_observation_configs()
    configs = steering_export._canonicalize_actor_observations(
        training_configs,
        all_observation_configs=training_configs,
    )
    module = ObservationExportModule(configs, context, device="cpu")

    assert tuple(module.get_input_keys()) == steering_export.EXPECTED_OBS_CONTEXT_KEYS
    assert all("noisy" not in key for key in module.get_input_keys())
    assert context.historical.actions.shape == (2, 8, 29)

    values = [
        steering_export._resolve_attr_path(key, context)
        for key in module.get_input_keys()
    ]
    outputs = module(*values)

    assert module.get_output_keys() == list(steering_export.EXPECTED_ACTOR_OBS_KEYS)
    assert [tuple(value.shape) for value in outputs] == [
        (2, 512),
        (2, 232),
        (2, 5),
    ]
    assert sum(value.shape[-1] for value in outputs) == 749
    assert torch.equal(outputs[1], context.historical.actions.reshape(2, 232))
    assert all(torch.isfinite(value).all() for value in outputs)


def test_frozen_timing_keeps_isaaclab_200_over_4():
    timing = steering_export._frozen_timing(
        SimpleNamespace(sim=SimpleNamespace(fps=200, decimation=4))
    )
    assert timing == {
        "physics_fps": 200,
        "physics_dt": pytest.approx(0.005),
        "decimation": 4,
        "control_dt": pytest.approx(0.02),
        "policy_hz": pytest.approx(50.0),
    }

    with pytest.raises(ValueError, match="frozen to IsaacLab 200 Hz"):
        steering_export._frozen_timing(
            SimpleNamespace(sim=SimpleNamespace(fps=500, decimation=10))
        )


class _TinyLazyActor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_keys = list(config.in_keys)
        self.out_keys = ["mean_action"]
        self.linear = torch.nn.LazyLinear(config.num_out)

    def forward(self, tensordict):
        obs = torch.cat([tensordict[key] for key in self.in_keys], dim=-1)
        tensordict["mean_action"] = self.linear(obs)
        return tensordict


def test_reconstruct_actor_materializes_lazy_layers_and_strictly_loads_checkpoint(
    monkeypatch,
):
    actor_config = SimpleNamespace(
        _target_="tests.TinyLazyActor",
        in_keys=["state", "command"],
        num_out=3,
    )
    obs_td = TensorDict(
        {
            "state": torch.tensor([[1.0, 2.0, 3.0]]),
            "command": torch.tensor([[4.0, 5.0]]),
        },
        batch_size=[1],
    )

    source = _TinyLazyActor(actor_config).eval()
    with torch.no_grad():
        expected = source(obs_td.clone())["mean_action"]
    checkpoint = {
        "model": {
            f"_actor.{key}": value.clone()
            for key, value in source.state_dict().items()
        }
    }
    import protomotions.utils.hydra_replacement as hydra_replacement

    monkeypatch.setattr(
        hydra_replacement, "get_class", lambda _target: _TinyLazyActor
    )
    loaded = steering_export._reconstruct_actor(
        actor_config, checkpoint, obs_td
    )

    with torch.no_grad():
        actual = loaded(obs_td.clone())["mean_action"]
    torch.testing.assert_close(actual, expected)
    assert loaded.linear.in_features == 5

    with pytest.raises(ValueError, match="no '_actor.' weights"):
        steering_export._reconstruct_actor(
            actor_config, {"model": {"critic.weight": torch.ones(1)}}, obs_td
        )


def test_sidecar_describes_nine_inputs_history_pd_and_steering_contract():
    actual_inputs = [
        steering_export._sanitize(key)
        for key in steering_export.EXPECTED_ONNX_CONTEXT_KEYS
    ]
    mapping = dict(zip(actual_inputs, steering_export.EXPECTED_ONNX_CONTEXT_KEYS))
    joint_names = [f"joint_{index}" for index in range(29)]
    sidecar = steering_export._build_yaml(
        actual_input_names=actual_inputs,
        actual_output_names=list(steering_export.EXPECTED_ONNX_OUTPUTS),
        onnx_name_to_key=mapping,
        input_shapes=_input_shapes(),
        output_shapes={name: [1, 29] for name in steering_export.EXPECTED_ONNX_OUTPUTS},
        obs_context_keys=list(steering_export.EXPECTED_OBS_CONTEXT_KEYS),
        actor_obs_shapes={
            "noisy_historical_reduced_coords_obs": [1, 512],
            "previous_actions": [1, 232],
            "noisy_steering": [1, 5],
        },
        joint_names=joint_names,
        body_names=["torso_link"],
        anchor_body_name="torso_link",
        anchor_body_index=0,
        mjcf_path="mjcf/elf3_bxi.xml",
        timing=steering_export._frozen_timing(
            SimpleNamespace(sim=SimpleNamespace(fps=200, decimation=4))
        ),
        history_steps=8,
        pd_action_offset=[0.0] * 29,
        action_scale=[0.5] * 29,
        stiffness=[100.0] * 29,
        damping=[5.0] * 29,
        effort_limits=[50.0] * 29,
        pd_target_max_accel=None,
        steering_speed_min=0.0,
        steering_speed_max=1.5,
        enable_rand_facing=False,
        checkpoint="epoch_30000.ckpt",
        resolved_config="resolved_configs_inference.pt",
        validation_max_diffs={"batch_2.actions": 1e-7},
    )

    assert sidecar["type"] == "unified_pipeline"
    assert sidecar["task"] == "steering"
    assert sidecar["dt"] == pytest.approx(0.02)
    assert sidecar["metadata"]["actor_input_dim"] == 749
    assert len(sidecar["policy_inputs"]) == 9
    assert len(sidecar["policy_outputs"]) == 4
    assert sidecar["_runtime"]["elided_context_keys"] == [
        "historical.root_rot"
    ]
    assert sidecar["timing"]["physics_fps"] == 200
    assert sidecar["timing"]["decimation"] == 4
    assert sidecar["timing"]["history_offsets_seconds"] == [
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14,
        0.16,
    ]
    action_history = next(
        entry
        for entry in sidecar["policy_inputs"]
        if entry["key"] == "historical.actions"
    )
    assert action_history["output_key"] == "actions"
    assert action_history["representation"] == "raw_policy_action"
    assert action_history["include_current_value_in_history"] is False
    speed = next(
        entry
        for entry in sidecar["policy_inputs"]
        if entry["key"] == "steering.tar_speed"
    )
    assert speed["shape"] == [1]
    assert speed["maximum"] == pytest.approx(1.5)
    assert sidecar["control"]["joint_target_formula"] == (
        "pd_action_offset + action_scale * actions"
    )
    assert "equal to tar_dir" in sidecar["steering"]["deployment_rule"]


class _TinyUnified(torch.nn.Module):
    def __init__(self, offset, scale, stiffness, damping):
        super().__init__()
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)
        self.register_buffer("stiffness", stiffness)
        self.register_buffer("damping", damping)

    def forward(self, action):
        targets = self.offset + self.scale * action
        return (
            action,
            targets,
            self.stiffness.unsqueeze(0).expand_as(action),
            self.damping.unsqueeze(0).expand_as(action),
        )


class _NumpySession:
    def __init__(self, offset, scale, stiffness, damping, *, drift=0.0):
        self.offset = offset.numpy()
        self.scale = scale.numpy()
        self.stiffness = stiffness.numpy()
        self.damping = damping.numpy()
        self.drift = drift
        self.batch_sizes = []

    def run(self, output_names, inputs):
        assert output_names == list(steering_export.EXPECTED_ONNX_OUTPUTS)
        action = inputs["historical_actions"].astype(np.float32)
        self.batch_sizes.append(action.shape[0])
        return [
            action + self.drift,
            self.offset + self.scale * action,
            np.broadcast_to(self.stiffness, action.shape).copy(),
            np.broadcast_to(self.damping, action.shape).copy(),
        ]


def test_ort_validation_checks_dynamic_batch_and_pd_sidecar():
    offset = torch.linspace(-0.2, 0.2, 29)
    scale = torch.linspace(0.5, 1.0, 29)
    stiffness = torch.linspace(10.0, 20.0, 29)
    damping = torch.linspace(1.0, 2.0, 29)
    unified = _TinyUnified(offset, scale, stiffness, damping)
    sample = torch.randn(1, 29)
    session = _NumpySession(offset, scale, stiffness, damping)

    diffs = steering_export._run_ort_validation(
        session=session,
        unified=unified,
        sample_inputs=[sample],
        obs_input_keys=["historical.actions"],
        onnx_name_to_key={"historical_actions": "historical.actions"},
        actual_output_names=list(steering_export.EXPECTED_ONNX_OUTPUTS),
        pd_action_offset=offset.tolist(),
        action_scale=scale.tolist(),
        stiffness=stiffness.tolist(),
        damping=damping.tolist(),
    )

    assert session.batch_sizes == [1, 2]
    assert max(diffs.values()) == pytest.approx(0.0)

    drifting_session = _NumpySession(
        offset, scale, stiffness, damping, drift=1e-2
    )
    with pytest.raises(ValueError, match="PyTorch/ONNX mismatch"):
        steering_export._run_ort_validation(
            session=drifting_session,
            unified=unified,
            sample_inputs=[sample],
            obs_input_keys=["historical.actions"],
            onnx_name_to_key={"historical_actions": "historical.actions"},
            actual_output_names=list(steering_export.EXPECTED_ONNX_OUTPUTS),
            pd_action_offset=offset.tolist(),
            action_scale=scale.tolist(),
            stiffness=stiffness.tolist(),
            damping=damping.tolist(),
        )
