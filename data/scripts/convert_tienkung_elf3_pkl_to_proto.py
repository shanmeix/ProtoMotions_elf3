# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert TienKung-Lab ELF3 pickle trajectories to ProtoMotions.

The supported source schema is a dictionary containing:

* ``root_pos``: ``(frames, 3)`` floating-point array in metres.
* ``root_rot``: ``(frames, 4)`` floating-point quaternion array in ``xyzw``.
* ``dof_pos``: ``(frames, 29)`` floating-point array in the fixed ELF3 joint
  order declared by :data:`ELF3_JOINT_NAMES`, in radians.
* ``fps``: a finite positive scalar sampling rate.

Root positions are passed through unchanged. In particular, the TienKung
visualizer's 0.3-m display offset is not applied.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import typer

if __package__:
    from .convert_elf3_csv_to_proto import (
        ELF3_JOINT_NAMES,
        REPO_ROOT,
        convert_elf3_arrays_to_proto,
    )
else:
    from convert_elf3_csv_to_proto import (
        ELF3_JOINT_NAMES,
        REPO_ROOT,
        convert_elf3_arrays_to_proto,
    )


app = typer.Typer(pretty_exceptions_enable=False)

ELF3_BXI_MJCF_PATH = REPO_ROOT / "protomotions/data/assets/mjcf/elf3_bxi.xml"
MAX_PICKLE_BYTES = 2 * 1024**3
REQUIRED_KEYS = frozenset({"root_pos", "root_rot", "dof_pos", "fps"})
OPTIONAL_KEYS = frozenset({"dof_names", "link_body_list", "local_body_pos"})


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Unpickle only the small NumPy vocabulary used by the source files."""

    _ALLOWED_GLOBALS = frozenset(
        {
            ("numpy", "dtype"),
            ("numpy", "ndarray"),
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"forbidden pickle global {module}.{name}; only NumPy arrays are allowed"
            )
        return super().find_class(module, name)


@dataclass(frozen=True)
class TienKungElf3Motion:
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    dof_pos: np.ndarray
    fps: float


@dataclass(frozen=True)
class ConversionJob:
    input_file: Path
    output_file: Path


def _load_restricted_pickle(input_file: Path) -> Any:
    file_size = input_file.stat().st_size
    if file_size <= 0:
        raise ValueError(f"{input_file} is empty.")
    if file_size > MAX_PICKLE_BYTES:
        raise ValueError(
            f"{input_file} is {file_size} bytes; the safety limit is "
            f"{MAX_PICKLE_BYTES} bytes."
        )

    try:
        with input_file.open("rb") as stream:
            value = _RestrictedNumpyUnpickler(stream).load()
            if stream.read(1):
                raise ValueError(f"{input_file} contains trailing data after the pickle.")
    except (EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(
            f"{input_file} is not a supported NumPy-only TienKung pickle: {exc}"
        ) from exc
    return value


def _validate_float_array(
    input_file: Path,
    key: str,
    value: Any,
    num_columns: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{input_file}: {key!r} must be a NumPy array.")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(
            f"{input_file}: {key!r} must use a floating dtype; got {value.dtype}."
        )
    if value.ndim != 2 or value.shape[1] != num_columns:
        raise ValueError(
            f"{input_file}: {key!r} must have shape (frames, {num_columns}); "
            f"got {value.shape}."
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{input_file}: {key!r} contains NaN or infinite values.")
    return np.ascontiguousarray(value, dtype=np.float32)


def _validate_fps(input_file: Path, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{input_file}: 'fps' must be a scalar number; got {value!r}.")
    try:
        fps = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{input_file}: 'fps' must be a scalar number; got {value!r}."
        ) from exc
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(
            f"{input_file}: 'fps' must be finite and positive; got {fps!r}."
        )
    return fps


def load_tienkung_elf3_pkl(input_file: Path) -> TienKungElf3Motion:
    """Safely load and validate one TienKung ELF3 motion pickle."""
    input_file = Path(input_file).resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"TienKung motion file not found: {input_file}")
    if input_file.suffix.lower() != ".pkl":
        raise ValueError(f"TienKung motion file must end in .pkl: {input_file}")

    payload = _load_restricted_pickle(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{input_file}: top-level pickle value must be a dictionary.")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{input_file}: all dictionary keys must be strings.")

    keys = set(payload)
    missing = REQUIRED_KEYS - keys
    unknown = keys - REQUIRED_KEYS - OPTIONAL_KEYS
    if missing:
        raise ValueError(f"{input_file}: missing required keys: {sorted(missing)}.")
    if unknown:
        raise ValueError(f"{input_file}: unsupported keys: {sorted(unknown)}.")

    for metadata_key in ("link_body_list", "local_body_pos"):
        if metadata_key in payload and payload[metadata_key] is not None:
            raise ValueError(
                f"{input_file}: optional {metadata_key!r} must be None; "
                "body-local source data is not part of this converter's contract."
            )

    if "dof_names" in payload:
        source_names = payload["dof_names"]
        if not isinstance(source_names, (list, tuple)):
            raise ValueError(f"{input_file}: optional 'dof_names' must be a list.")
        if list(source_names) != ELF3_JOINT_NAMES:
            raise ValueError(
                f"{input_file}: 'dof_names' does not match the ELF3 contract.\n"
                f"source: {list(source_names)}\nexpected: {ELF3_JOINT_NAMES}"
            )

    root_pos = _validate_float_array(input_file, "root_pos", payload["root_pos"], 3)
    root_quat_xyzw = _validate_float_array(
        input_file, "root_rot", payload["root_rot"], 4
    )
    dof_pos = _validate_float_array(input_file, "dof_pos", payload["dof_pos"], 29)
    fps = _validate_fps(input_file, payload["fps"])

    num_frames = root_pos.shape[0]
    if num_frames < 2:
        raise ValueError(f"{input_file}: at least two frames are required.")
    if root_quat_xyzw.shape[0] != num_frames or dof_pos.shape[0] != num_frames:
        raise ValueError(
            f"{input_file}: frame counts differ: root_pos={num_frames}, "
            f"root_rot={root_quat_xyzw.shape[0]}, dof_pos={dof_pos.shape[0]}."
        )

    quat_norm = np.linalg.norm(root_quat_xyzw, axis=-1, keepdims=True)
    if np.any(quat_norm < 1.0e-8):
        raise ValueError(f"{input_file}: 'root_rot' contains a zero-length quaternion.")
    root_quat_xyzw = root_quat_xyzw / quat_norm
    root_quat_wxyz = np.ascontiguousarray(root_quat_xyzw[:, [3, 0, 1, 2]])

    return TienKungElf3Motion(
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        dof_pos=dof_pos,
        fps=fps,
    )


def build_conversion_jobs(input_path: Path, output_dir: Path) -> List[ConversionJob]:
    """Map one pkl or a recursive pkl tree into an output directory."""
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pkl":
            raise ValueError(f"Input file must end in .pkl: {input_path}")
        input_files = [input_path]
        input_root = input_path.parent
    elif input_path.is_dir():
        input_files = sorted(
            path.resolve()
            for path in input_path.rglob("*.pkl")
            if path.is_file()
        )
        input_root = input_path
        if not input_files:
            raise FileNotFoundError(f"No .pkl files found below {input_path}")
    else:
        raise ValueError(
            f"Input path is neither a regular file nor directory: {input_path}"
        )

    return [
        ConversionJob(
            input_file=input_file,
            output_file=(output_dir / input_file.relative_to(input_root)).with_suffix(
                ".motion"
            ),
        )
        for input_file in input_files
    ]


def convert_tienkung_dataset(
    input_path: Path,
    output_dir: Path,
    mjcf_path: Path = ELF3_BXI_MJCF_PATH,
    contact_velocity_threshold: float = 0.15,
    contact_height_threshold: float = 0.1,
    force_remake: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Convert one file or a recursive directory, returning converted/skipped paths."""
    mjcf_path = Path(mjcf_path).resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"ELF3 BXI MJCF file not found: {mjcf_path}")

    converted: List[Path] = []
    skipped: List[Path] = []
    for job in build_conversion_jobs(input_path, output_dir):
        if job.output_file.exists() and not job.output_file.is_file():
            raise IsADirectoryError(
                f"Output motion path is not a regular file: {job.output_file}"
            )
        if job.output_file.exists() and not force_remake:
            print(f"Skipping existing {job.output_file}")
            skipped.append(job.output_file)
            continue

        source = load_tienkung_elf3_pkl(job.input_file)
        motion = convert_elf3_arrays_to_proto(
            root_pos_np=source.root_pos,
            root_quat_wxyz_np=source.root_quat_wxyz,
            joint_pos_np=source.dof_pos,
            output_file=job.output_file,
            fps=source.fps,
            mjcf_path=mjcf_path,
            contact_velocity_threshold=contact_velocity_threshold,
            contact_height_threshold=contact_height_threshold,
            force_remake=force_remake,
        )
        converted.append(job.output_file)
        print(f"Converted {job.input_file}")
        print(f"  fps:             {motion.fps}")
        print(f"  dof_pos:         {tuple(motion.dof_pos.shape)}")
        print(f"  rigid_body_pos:  {tuple(motion.rigid_body_pos.shape)}")
        print(f"  contacts:        {tuple(motion.rigid_body_contacts.shape)}")
        print(f"  saved:           {job.output_file}")

    return converted, skipped


@app.command()
def main(
    input_path: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=True,
        help="One TienKung ELF3 .pkl file or a directory searched recursively.",
    ),
    output_dir: Path = typer.Option(
        ...,
        file_okay=False,
        dir_okay=True,
        help="Output directory; recursive input layout is preserved.",
    ),
    mjcf_path: Path = typer.Option(
        ELF3_BXI_MJCF_PATH,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help="ELF3 MJCF used for FK and joint-limit validation.",
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
        help="Overwrite output motions instead of safely skipping them.",
    ),
):
    """Convert TienKung ELF3 pkl data with its embedded per-file fps."""
    converted, skipped = convert_tienkung_dataset(
        input_path=input_path,
        output_dir=output_dir,
        mjcf_path=mjcf_path,
        contact_velocity_threshold=contact_velocity_threshold,
        contact_height_threshold=contact_height_threshold,
        force_remake=force_remake,
    )
    print(f"Finished: converted={len(converted)}, skipped={len(skipped)}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
