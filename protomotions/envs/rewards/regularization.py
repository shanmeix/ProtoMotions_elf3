# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regularization reward compute kernels.

Pure tensor functions (kernels) for computing regularization rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.regularization import compute_action_smoothness
    
    reward_components = {
        "action_smoothness": MdpComponent(
            compute_func=compute_action_smoothness,
            dynamic_vars={
                "current_processed_action": EnvContext.current_processed_action,
                "previous_processed_action": EnvContext.previous_processed_action,
            },
        ),
    }

Includes:
- Action smoothness (L2 and Log-Mean-Exp variants)
- Power consumption
- Joint limit violations
- Contact matching
- Contact force change penalties
"""

import math
from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.rewards.base import (
    delta_logmeanexp,
    delta_norm,
    power_consumption_sum,
)
from protomotions.utils import rotations


# =============================================================================
# Regularization Reward Kernels
# =============================================================================

def compute_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
) -> Tensor:
    """Action smoothness reward (L2 norm of processed action changes).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_norm(current_processed_action, previous_processed_action)


def compute_action_smoothness_logmeanexp(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    beta: float = 3.0,
) -> Tensor:
    """Action smoothness using Log-Mean-Exp (soft L_infinity).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
        beta: Temperature parameter. Lower = more like mean, higher = more like max.
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_logmeanexp(
        current_processed_action,
        previous_processed_action,
        beta=beta,
    )


def compute_action_rate_l2(
    current_action: Tensor,
    previous_action: Tensor,
    joint_indices: Optional[Tensor] = None,
) -> Tensor:
    """Squared raw-action change, matching the TienKung locomotion penalty."""
    action_delta = current_action - previous_action
    if joint_indices is not None:
        action_delta = action_delta[:, joint_indices]
    return torch.sum(torch.square(action_delta), dim=-1)


def compute_action_l1(
    action: Tensor,
    joint_indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum absolute raw actions, optionally over a selected joint subset."""
    if joint_indices is not None:
        action = action[:, joint_indices]
    return torch.sum(torch.abs(action), dim=-1)


def compute_pow_rew(
    dof_forces: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
    joint_indices: Optional[Tensor] = None,
) -> Tensor:
    """Power consumption reward.
    
    Args:
        dof_forces: Joint forces/torques [num_envs, num_dofs].
        dof_vel: Joint velocities [num_envs, num_dofs].
        use_torque_squared: Whether to use torque squared instead of absolute.
        joint_indices: Optional joint subset to include.
    
    Returns:
        Power consumption tensor [num_envs].
    """
    return power_consumption_sum(
        dof_forces,
        dof_vel,
        use_torque_squared,
        indices=joint_indices,
    )


def compute_soft_pos_limit_rew(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    max_violation: float = None,
) -> Tensor:
    """Soft joint position limit penalty.

    Penalizes when joints approach or exceed limits.

    The penalty is linear in the violation and summed over every DOF, so it has
    no floor. That is harmless while the simulator is healthy and unbounded when
    it is not: joint positions that blow up produce an arbitrarily large negative
    reward, which then sets the reward-normalizer scale for many epochs
    afterwards. A raw task reward of -7.9e7 was traced to this term.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower joint limits [num_dofs].
        dof_limits_upper: Upper joint limits [num_dofs].
        max_violation: If set, clamp the summed violation (in radians) before it
            is weighted, which puts a floor on the penalty. Past the clamp the
            state is broken rather than merely bad, and a larger number carries
            no useful gradient. ``None`` (default) keeps the unbounded behaviour.

    Returns:
        Penalty tensor [num_envs].
    """
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    total = torch.sum(out_of_limits, dim=1)
    if max_violation is not None:
        total = total.clip(max=max_violation)
    return total


def compute_joint_deviation_l1(
    dof_pos: Tensor,
    default_dof_pos: Tensor,
    joint_indices: Optional[Tensor] = None,
) -> Tensor:
    """L1 joint deviation from a fixed default pose."""
    deviation = dof_pos - default_dof_pos
    if joint_indices is not None:
        deviation = deviation[:, joint_indices]
    return torch.sum(torch.abs(deviation), dim=-1)


def compute_zero_command_joint_deviation_rew(
    dof_pos: Tensor,
    default_dof_pos: Tensor,
    tar_speed: Tensor,
    anchor_rot: Tensor,
    tar_face_dir: Tensor,
    joint_indices: Optional[Tensor] = None,
    speed_threshold: float = 0.1,
    facing_tolerance: float = 0.2,
) -> Tensor:
    """Default-pose deviation when stopped and already facing the command.

    Steering can request an in-place heading change while target speed is zero.
    The facing gate avoids fighting that maneuver with a standing-pose penalty.
    """
    deviation = compute_joint_deviation_l1(
        dof_pos,
        default_dof_pos,
        joint_indices=joint_indices,
    )
    face_dir_3d = torch.cat(
        [tar_face_dir, torch.zeros_like(tar_face_dir[..., :1])], dim=-1
    )
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last=True)
    local_face_dir = rotations.quat_rotate(heading_inv, face_dir_3d, w_last=True)
    stopped = torch.abs(tar_speed) < speed_threshold
    facing_aligned = local_face_dir[..., 0] > math.cos(facing_tolerance)
    active = torch.logical_and(stopped, facing_aligned).to(deviation.dtype)
    return deviation * active


def compute_feet_y_distance_rew(
    rigid_body_pos: Tensor,
    anchor_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    left_foot_body_index: int,
    right_foot_body_index: int,
    target_distance: float = 0.299,
    lateral_speed_threshold: float = 0.1,
) -> Tensor:
    """Foot lateral-spacing error when little sideways motion is commanded."""
    heading_inv = rotations.calc_heading_quat_inv(anchor_rot, w_last=True)

    feet_delta = (
        rigid_body_pos[:, left_foot_body_index]
        - rigid_body_pos[:, right_foot_body_index]
    )
    local_feet_delta = rotations.quat_rotate(
        heading_inv, feet_delta, w_last=True
    )
    lateral_distance = torch.abs(local_feet_delta[..., 1])

    tar_dir_3d = torch.cat(
        [tar_dir, torch.zeros_like(tar_dir[..., :1])], dim=-1
    )
    local_tar_dir = rotations.quat_rotate(
        heading_inv, tar_dir_3d, w_last=True
    )
    lateral_speed = torch.abs(local_tar_dir[..., 1] * tar_speed)
    active = (lateral_speed < lateral_speed_threshold).to(lateral_distance.dtype)
    return torch.abs(lateral_distance - target_distance) * active


def compute_contact_match_rew(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    contact_body_ids: Tensor,
) -> Tensor:
    """Contact matching reward using foot contact bodies.
    
    Penalizes mismatch between simulated and reference foot contacts.
    Uses contact_body_ids (typically foot bodies).
    
    Args:
        sim_contacts: Simulated contact flags [num_envs, num_bodies].
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        contact_body_ids: Indices of bodies to track contacts for [num_contact_bodies].
    
    Returns:
        Contact mismatch penalty tensor [num_envs].
    """
    sim_contacts_subset = sim_contacts[:, contact_body_ids]
    ref_contacts_subset = ref_contacts[:, contact_body_ids]
    return torch.abs(sim_contacts_subset.float() - ref_contacts_subset.float()).sum(dim=1)


def compute_contact_force_change_rew(
    current_contact_force_magnitudes: Tensor,
    prev_contact_force_magnitudes: Tensor,
    threshold: float = 30.0,
) -> Tensor:
    """Contact force change penalty.
    
    Penalizes sudden contact force changes above a threshold (impact penalty).
    
    Args:
        current_contact_force_magnitudes: Current contact forces [num_envs, num_bodies].
        prev_contact_force_magnitudes: Previous contact forces [num_envs, num_bodies].
        threshold: Force change threshold below which changes are ignored (default: 30.0).
    
    Returns:
        Total force change above threshold [num_envs].
    """
    force_changes = torch.abs(current_contact_force_magnitudes - prev_contact_force_magnitudes)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


# =============================================================================
# Helper Functions (used by kernels or for advanced use cases)
# =============================================================================

def joint_limit_violation(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of joint position limit violations.

    Penalizes positions outside [lower, upper] limits.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower limits [num_dofs].
        dof_limits_upper: Upper limits [num_dofs].
        indices: Optional DOF indices to subset.

    Returns:
        Total violation [num_envs].
    """
    if indices is not None:
        dof_pos = dof_pos[:, indices]
        dof_limits_lower = dof_limits_lower[indices]
        dof_limits_upper = dof_limits_upper[indices]

    below_lower = -(dof_pos - dof_limits_lower).clip(max=0.0)
    above_upper = (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(below_lower + above_upper, dim=1)


def contact_mismatch_sum(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of contact state mismatches.

    Computes sum(|sim_contacts - ref_contacts|).

    Args:
        sim_contacts: Simulated contacts [num_envs, num_bodies].
        ref_contacts: Reference contacts [num_envs, num_bodies].
        indices: Optional body indices to subset.

    Returns:
        Total mismatch [num_envs].
    """
    if indices is not None:
        sim_contacts = sim_contacts[:, indices]
        ref_contacts = ref_contacts[:, indices]

    return torch.abs(sim_contacts.float() - ref_contacts.float()).sum(dim=1)


def impact_force_penalty(
    current_forces: Tensor,
    previous_forces: Tensor,
    indices: Optional[Tensor] = None,
    threshold: float = 30.0,
) -> Tensor:
    """Sum of sudden contact force changes above a threshold (impact penalty).

    Penalizes abrupt force changes (both increases and decreases) that exceed
    the threshold. Small force changes below the threshold are ignored.

    Args:
        current_forces: Current contact forces [num_envs, num_bodies].
        previous_forces: Previous contact forces [num_envs, num_bodies].
        indices: Optional body indices to subset.
        threshold: Force change threshold below which changes are ignored (default: 30.0).

    Returns:
        Total force change above threshold [num_envs].
    """
    if indices is not None:
        current_forces = current_forces[:, indices]
        previous_forces = previous_forces[:, indices]

    force_changes = torch.abs(current_forces - previous_forces)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


__all__ = [
    # Main reward kernels
    "compute_action_smoothness",
    "compute_action_smoothness_logmeanexp",
    "compute_action_rate_l2",
    "compute_action_l1",
    "compute_pow_rew",
    "compute_soft_pos_limit_rew",
    "compute_joint_deviation_l1",
    "compute_zero_command_joint_deviation_rew",
    "compute_feet_y_distance_rew",
    "compute_contact_match_rew",
    "compute_contact_force_change_rew",
    # Helper functions
    "joint_limit_violation",
    "contact_mismatch_sum",
    "impact_force_penalty",
]
