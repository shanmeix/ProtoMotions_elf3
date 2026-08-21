"""Regression tests for the simulator-free BeyondMimic export context."""

import torch

from deployment.export_bm_tracker_onnx import (
    MockContext,
    _canonicalize_deployment_observation_configs,
)
from protomotions.envs.component_factories import (
    mimic_target_poses_reduced_coords_factory,
    previous_actions_factory,
    reduced_coords_obs_factory,
)
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.utils.export_utils import (
    ObservationExportModule,
    _resolve_context_path,
)


def test_bm_mock_context_supports_noisy_actor_observations():
    """Training noisy bindings become canonical single-state deployment inputs.

    BeyondMimic trains its actor with ``EnvContext.noisy`` paths.  At export
    and deployment there is only one measured state input, so those bindings
    must use ``current`` without mutating the frozen training configuration.
    """

    context = MockContext(
        num_envs=2,
        num_dofs=29,
        num_bodies=30,
        num_future_steps=4,
        anchor_idx=0,
        history_steps=1,
    )

    all_training_configs = {
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "noisy_mimic_reduced_coords_target_poses": (
            mimic_target_poses_reduced_coords_factory(
                use_noisy=True,
                include_dof_vel=True,
                include_xy_offset=False,
            )
        ),
        "clean_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "clean_mimic_reduced_coords_target_poses": (
            mimic_target_poses_reduced_coords_factory(
                use_noisy=False,
                include_dof_vel=True,
                include_xy_offset=False,
            )
        ),
        "historical_previous_processed_actions": previous_actions_factory(
            history_steps=1,
            processed=True,
        ),
    }
    actor_keys = (
        "noisy_reduced_coords_obs",
        "noisy_mimic_reduced_coords_target_poses",
        "historical_previous_processed_actions",
    )
    actor_configs = {key: all_training_configs[key] for key in actor_keys}
    obs_pairs = {
        "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
        "noisy_mimic_reduced_coords_target_poses": (
            "clean_mimic_reduced_coords_target_poses"
        ),
    }
    observation_configs = _canonicalize_deployment_observation_configs(
        actor_configs,
        all_observation_configs=all_training_configs,
        obs_pairs=obs_pairs,
    )

    assert observation_configs["noisy_reduced_coords_obs"] is not all_training_configs[
        "noisy_reduced_coords_obs"
    ]
    assert observation_configs["noisy_reduced_coords_obs"] is not all_training_configs[
        "clean_reduced_coords_obs"
    ]
    assert observation_configs["noisy_reduced_coords_obs"].static_params is (
        all_training_configs["clean_reduced_coords_obs"].static_params
    )
    assert all_training_configs[
        "noisy_reduced_coords_obs"
    ].get_bindings_dict()["dof_pos"] == "noisy.dof_pos"
    assert observation_configs[
        "noisy_reduced_coords_obs"
    ].get_bindings_dict()["dof_pos"] == "current.dof_pos"

    module = ObservationExportModule(
        observation_configs,
        context,
        device=torch.device("cpu"),
    )
    input_keys = module.get_input_keys()
    assert "current.anchor_rot" in input_keys
    assert "current.dof_pos" in input_keys
    assert "current.dof_vel" in input_keys
    assert "current.root_local_ang_vel" in input_keys
    assert all("noisy" not in path for path in input_keys)
    inputs = [
        _resolve_context_path(path, context) for path in input_keys
    ]

    outputs = module(*inputs)

    assert module.get_output_keys() == list(observation_configs)
    assert len(outputs) == 3
    assert all(output.shape[0] == 2 for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_bm_deployment_canonicalization_covers_all_noisy_context_views():
    component = MdpComponent(
        compute_func=lambda **kwargs: kwargs["current"],
        dynamic_vars={
            "current": EnvContext.current.dof_pos,
            "noisy": EnvContext.noisy.dof_pos,
            "noisy_history": EnvContext.noisy_historical.dof_pos,
            "noisy_ground": EnvContext.noisy_ground_heights,
        },
    )

    canonical = _canonicalize_deployment_observation_configs(
        {"component": component}
    )["component"]

    assert canonical is not component
    assert canonical.static_params is component.static_params
    assert component.get_bindings_dict() == {
        "current": "current.dof_pos",
        "noisy": "noisy.dof_pos",
        "noisy_history": "noisy_historical.dof_pos",
        "noisy_ground": "noisy_ground_heights",
    }
    assert canonical.get_bindings_dict() == {
        "current": "current.dof_pos",
        "noisy": "current.dof_pos",
        "noisy_history": "historical.dof_pos",
        "noisy_ground": "ground_heights",
    }


def test_bm_deployment_rejects_a_missing_l2c2_clean_component():
    noisy_component = reduced_coords_obs_factory(use_noisy=True)

    try:
        _canonicalize_deployment_observation_configs(
            {"noisy_obs": noisy_component},
            all_observation_configs={"noisy_obs": noisy_component},
            obs_pairs={"noisy_obs": "clean_obs"},
        )
    except ValueError as error:
        assert "missing clean observation component 'clean_obs'" in str(error)
    else:
        raise AssertionError("missing L2C2 clean component was accepted")
