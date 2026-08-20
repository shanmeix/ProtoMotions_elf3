# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert an elf3_mjlab CSV trajectory to ProtoMotions ``.motion`` format.

The ELF3 CSV contract is:

``root_position_xyz[m], root_quaternion_xyzw, 29 joint_positions[rad]``

The joint columns must follow the depth-first MJCF order encoded below. Root
positions are preserved exactly; in particular, this converter does not apply
the height correction used by the retargeted G1 converter.
"""

from pathlib import Path

import numpy as np
import torch
import typer

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)

if __package__:
    from .contact_detection import compute_contact_labels_from_pos_and_vel
else:
    from contact_detection import compute_contact_labels_from_pos_and_vel


app = typer.Typer(pretty_exceptions_enable=False)

REPO_ROOT = Path(__file__).resolve().parents[2]
ELF3_MJCF_PATH = REPO_ROOT / "protomotions/data/assets/mjcf/elf3.xml"

ELF3_JOINT_NAMES = [
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


def load_elf3_csv(
    input_file: Path,
    ignore_first_n_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate an ELF3 CSV, returning xyz, wxyz and joint arrays."""
    data = np.loadtxt(input_file, delimiter=",", dtype=np.float32)
    data = np.atleast_2d(data)

    expected_columns = 7 + len(ELF3_JOINT_NAMES)
    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise ValueError(
            f"{input_file} must have {expected_columns} columns "
            f"(xyz + quaternion xyzw + {len(ELF3_JOINT_NAMES)} joints); "
            f"got shape {data.shape}."
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{input_file} contains NaN or infinite values.")
    if ignore_first_n_frames < 0:
        raise ValueError("--ignore-first-n-frames must be non-negative.")
    if data.shape[0] - ignore_first_n_frames < 2:
        raise ValueError("At least two frames must remain after dropping frames.")

    data = data[ignore_first_n_frames:]
    root_pos = data[:, :3]
    root_quat_xyzw = data[:, 3:7]
    joint_pos = data[:, 7:]

    quat_norm = np.linalg.norm(root_quat_xyzw, axis=-1, keepdims=True)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError(f"{input_file} contains a zero-length root quaternion.")
    root_quat_xyzw = root_quat_xyzw / quat_norm
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    return root_pos, root_quat_wxyz, joint_pos


@app.command()
def main(
    input_file: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Headerless elf3_mjlab CSV file.",
    ),
    output_file: Path = typer.Option(
        ...,
        help="Destination ProtoMotions .motion file.",
    ),
    fps: int = typer.Option(30, min=1, help="Sampling rate of the input CSV."),
    ignore_first_n_frames: int = typer.Option(
        0,
        min=0,
        help="Drop this many leading frames before FK and velocity calculation.",
    ),
    contact_velocity_threshold: float = typer.Option(
        0.15,
        min=0.0,
        help="Maximum body speed (m/s) for heuristic contact labels.",
    ),
    contact_height_threshold: float = typer.Option(
        0.1,
        help="Maximum body height (m) for heuristic contact labels.",
    ),
    force_remake: bool = typer.Option(
        False,
        "--force-remake",
        help="Overwrite output_file when it already exists.",
    ),
):
    """Convert one ELF3 CSV trajectory without changing its root trajectory."""
    input_file = input_file.resolve()
    output_file = output_file.resolve()
    if output_file.suffix != ".motion":
        raise ValueError("--output-file must end in .motion")
    if output_file.exists() and not force_remake:
        raise FileExistsError(
            f"{output_file} already exists; pass --force-remake to overwrite it."
        )

    kinematic_info = extract_kinematic_info(str(ELF3_MJCF_PATH))
    if kinematic_info.dof_names != ELF3_JOINT_NAMES:
        raise ValueError(
            "ELF3 MJCF DOF order no longer matches the CSV contract:\n"
            f"CSV:  {ELF3_JOINT_NAMES}\nMJCF: {kinematic_info.dof_names}"
        )

    root_pos_np, root_quat_wxyz_np, joint_pos_np = load_elf3_csv(
        input_file,
        ignore_first_n_frames=ignore_first_n_frames,
    )
    root_pos = torch.from_numpy(root_pos_np)
    root_quat_wxyz = torch.from_numpy(root_quat_wxyz_np)
    joint_pos = torch.from_numpy(joint_pos_np)

    lower = kinematic_info.dof_limits_lower.to(joint_pos)
    upper = kinematic_info.dof_limits_upper.to(joint_pos)
    tolerance = 1.0e-5
    if ((joint_pos < lower - tolerance) | (joint_pos > upper + tolerance)).any():
        violating = torch.nonzero(
            (joint_pos < lower - tolerance) | (joint_pos > upper + tolerance),
            as_tuple=False,
        )[0]
        frame_idx, dof_idx = violating.tolist()
        raise ValueError(
            f"Joint limit violation at output frame {frame_idx}, "
            f"{ELF3_JOINT_NAMES[dof_idx]}={joint_pos[frame_idx, dof_idx].item():.6f}; "
            f"expected [{lower[dof_idx].item():.6f}, {upper[dof_idx].item():.6f}]."
        )

    qpos = torch.cat([root_pos, root_quat_wxyz, joint_pos], dim=-1)
    fk_root_pos, joint_rot_mats = extract_transforms_from_qpos(
        kinematic_info,
        qpos,
    )
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=fk_root_pos,
        joint_rot_mats=joint_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.dof_pos = joint_pos
    motion.dof_vel = compute_cartesian_velocity(
        joint_pos.unsqueeze(1),
        fps=fps,
        velocity_max_horizon=1,
    ).squeeze(1)
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=contact_velocity_threshold,
        height_thresh=contact_height_threshold,
    ).to(torch.bool)

    # ELF3 has one hinge DOF per non-root body. MotionLib must not interpret
    # these rotations as 3-DOF exponential-map joint data during interpolation.
    motion.local_rigid_body_rot = None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), output_file)

    print(f"Converted {input_file}")
    print(f"  skipped frames:  {ignore_first_n_frames}")
    print(f"  fps:             {motion.fps}")
    print(f"  dof_pos:         {tuple(motion.dof_pos.shape)}")
    print(f"  rigid_body_pos:  {tuple(motion.rigid_body_pos.shape)}")
    print(f"  contacts:        {tuple(motion.rigid_body_contacts.shape)}")
    print(f"  saved:           {output_file}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
