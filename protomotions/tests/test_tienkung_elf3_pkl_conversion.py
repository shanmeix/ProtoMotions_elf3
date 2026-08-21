# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.scripts import convert_tienkung_elf3_pkl_to_proto as converter


def _valid_payload(num_frames=4):
    root_pos = np.arange(num_frames * 3, dtype=np.float64).reshape(num_frames, 3)
    root_pos[:, 2] = 0.95
    root_rot = np.tile(
        np.array([[0.0, 0.0, 0.0, 2.0]], dtype=np.float64),
        (num_frames, 1),
    )
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": np.zeros((num_frames, 29), dtype=np.float64),
        "fps": np.float64(29.97),
        "link_body_list": None,
        "local_body_pos": None,
    }


def _write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)


def test_load_tienkung_pickle_validates_xyzw_and_preserves_root_height(tmp_path):
    input_file = tmp_path / "walk.pkl"
    payload = _valid_payload()
    _write_pickle(input_file, payload)

    motion = converter.load_tienkung_elf3_pkl(input_file)

    assert motion.root_pos.dtype == np.float32
    np.testing.assert_allclose(motion.root_pos, payload["root_pos"])
    np.testing.assert_allclose(motion.root_pos[:, 2], 0.95)
    np.testing.assert_allclose(
        motion.root_quat_wxyz,
        np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1)),
    )
    assert motion.dof_pos.shape == (4, 29)
    assert motion.fps == pytest.approx(29.97)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("fps"), "missing required keys"),
        (
            lambda payload: payload.__setitem__(
                "dof_pos", np.zeros((4, 28), dtype=np.float32)
            ),
            r"dof_pos.*\(frames, 29\)",
        ),
        (lambda payload: payload.__setitem__("fps", 0.0), "finite and positive"),
        (lambda payload: payload.__setitem__("fps", "30"), "scalar number"),
        (
            lambda payload: payload.__setitem__("unexpected", None),
            "unsupported keys",
        ),
        (
            lambda payload: payload.__setitem__(
                "dof_names", list(reversed(converter.ELF3_JOINT_NAMES))
            ),
            "does not match the ELF3 contract",
        ),
    ],
)
def test_load_tienkung_pickle_rejects_invalid_schema(tmp_path, mutate, message):
    input_file = tmp_path / "bad.pkl"
    payload = _valid_payload()
    mutate(payload)
    _write_pickle(input_file, payload)

    with pytest.raises(ValueError, match=message):
        converter.load_tienkung_elf3_pkl(input_file)


def test_restricted_unpickler_rejects_executable_global(tmp_path):
    marker = tmp_path / "unsafe-code-ran"

    class UnsafePayload:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    input_file = tmp_path / "unsafe.pkl"
    _write_pickle(input_file, UnsafePayload())

    with pytest.raises(ValueError, match="forbidden pickle global"):
        converter.load_tienkung_elf3_pkl(input_file)
    assert not marker.exists()


def test_build_conversion_jobs_supports_file_and_recursive_directory(tmp_path):
    input_dir = tmp_path / "input"
    first = input_dir / "stand.pkl"
    second = input_dir / "nested" / "walk.pkl"
    _write_pickle(first, _valid_payload())
    _write_pickle(second, _valid_payload())
    output_dir = tmp_path / "output"

    directory_jobs = converter.build_conversion_jobs(input_dir, output_dir)
    assert [(job.input_file, job.output_file) for job in directory_jobs] == [
        (second.resolve(), output_dir / "nested" / "walk.motion"),
        (first.resolve(), output_dir / "stand.motion"),
    ]

    file_jobs = converter.build_conversion_jobs(second, output_dir)
    assert [(job.input_file, job.output_file) for job in file_jobs] == [
        (second.resolve(), output_dir / "walk.motion")
    ]


def test_convert_dataset_uses_bxi_fps_and_force_remake(tmp_path, monkeypatch):
    input_file = tmp_path / "walk.pkl"
    payload = _valid_payload()
    _write_pickle(input_file, payload)
    output_dir = tmp_path / "output"
    calls = []

    def fake_convert(**kwargs):
        calls.append(kwargs)
        output_file = kwargs["output_file"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"motion")
        frames = kwargs["joint_pos_np"].shape[0]
        return SimpleNamespace(
            fps=kwargs["fps"],
            dof_pos=torch.zeros((frames, 29)),
            rigid_body_pos=torch.zeros((frames, 30, 3)),
            rigid_body_contacts=torch.zeros((frames, 30), dtype=torch.bool),
        )

    monkeypatch.setattr(converter, "convert_elf3_arrays_to_proto", fake_convert)

    converted, skipped = converter.convert_tienkung_dataset(input_file, output_dir)
    assert converted == [output_dir / "walk.motion"]
    assert skipped == []
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["root_pos_np"], payload["root_pos"])
    assert calls[0]["fps"] == pytest.approx(29.97)
    assert calls[0]["mjcf_path"].name == "elf3_bxi.xml"

    converted, skipped = converter.convert_tienkung_dataset(input_file, output_dir)
    assert converted == []
    assert skipped == [output_dir / "walk.motion"]
    assert len(calls) == 1

    converted, skipped = converter.convert_tienkung_dataset(
        input_file,
        output_dir,
        force_remake=True,
    )
    assert converted == [output_dir / "walk.motion"]
    assert skipped == []
    assert len(calls) == 2
    assert calls[-1]["force_remake"] is True
