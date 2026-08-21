#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the 29-DOF ProtoMotions ELF3 asset from the BXI MuJoCo model.

The BXI simulator model has two actuated head bodies in addition to the
30-body, 29-DOF policy skeleton used by ProtoMotions. This script locks both head
joints at zero, moves their visual/collision geoms onto ``torso_link``, and
combines the three inertials using rigid-body composition and the parallel-axis
theorem.  Cameras, lights, sensors, lidar and terrain includes are intentionally
not copied into the robot-only output.

The source layout is validated before conversion so that a future BXI model
change cannot silently alter the policy body/joint order.

Usage::

    python data/scripts/build_elf3_bxi_mjcf.py
    python data/scripts/build_elf3_bxi_mjcf.py --check
    python data/scripts/build_elf3_bxi_mjcf.py --source /path/to/elf3.xml
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT.parent
    / "bxi_rl_controller_ros2_example"
    / "src/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml"
)
DEFAULT_OUTPUT = REPO_ROOT / "protomotions/data/assets/mjcf/elf3_bxi.xml"
DEFAULT_MESH_OUTPUT_DIR = REPO_ROOT / "protomotions/data/assets/mesh/ELF3"

POLICY_BODY_NAMES = (
    "torso_link",
    "waist_y_link",
    "waist_x_link",
    "waist_z_link",
    "l_hip_y_link",
    "l_hip_x_link",
    "l_hip_z_link",
    "l_knee_y_link",
    "l_ankle_y_link",
    "l_ankle_x_link",
    "r_hip_y_link",
    "r_hip_x_link",
    "r_hip_z_link",
    "r_knee_y_link",
    "r_ankle_y_link",
    "r_ankle_x_link",
    "l_shoulder_y_link",
    "l_shoulder_x_link",
    "l_shoulder_z_link",
    "l_elbow_y_link",
    "l_wrist_x_link",
    "l_wrist_y_link",
    "l_wrist_z_link",
    "r_shoulder_y_link",
    "r_shoulder_x_link",
    "r_shoulder_z_link",
    "r_elbow_y_link",
    "r_wrist_x_link",
    "r_wrist_y_link",
    "r_wrist_z_link",
)

POLICY_JOINT_NAMES = (
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
)

HEAD_BODY_NAMES = ("head_z_link", "head_y_link")
HEAD_JOINT_NAMES = ("head_z_joint", "head_y_joint")

# Most BXI meshes are byte-identical to the existing ELF3 assets.  Keep using
# those shared files, but give changed files distinct names so the legacy
# ``elf3.xml`` remains reproducible.  The head meshes do not exist in ELF3-lite
# and are copied under their original names.
COPIED_MESH_FILES = {
    "torso_link.STL": "torso_link_bxi.STL",
    "l_ankle_x_link.STL": "l_ankle_x_link_bxi.STL",
    "r_ankle_x_link.STL": "r_ankle_x_link_bxi.STL",
    "head_z_link.STL": "head_z_link.STL",
    "head_y_link.STL": "head_y_link.STL",
}

Vector = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Matrix = Tuple[Vector, Vector, Vector]
Transform = Tuple[Vector, Quaternion]

ZERO: Vector = (0.0, 0.0, 0.0)
IDENTITY_QUAT: Quaternion = (1.0, 0.0, 0.0, 0.0)
IDENTITY_MATRIX: Matrix = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class MassProperties:
    mass: float
    com: Vector
    inertia_at_com: Matrix


@dataclass(frozen=True)
class BuildResult:
    xml_bytes: bytes
    copied_meshes: Mapping[Path, Path]
    torso: MassProperties
    source_mass: float
    source_com: Vector
    source_inertia: Matrix
    output_mass: float
    output_com: Vector
    output_inertia: Matrix


def _fail(message: str) -> None:
    raise ValueError(message)


def _floats(
    value: Optional[str], count: int, default: Sequence[float]
) -> Tuple[float, ...]:
    if value is None:
        values = tuple(float(item) for item in default)
    else:
        values = tuple(float(item) for item in value.split())
    if len(values) != count:
        _fail(f"Expected {count} numbers, got {len(values)} in {value!r}")
    return values


def _vector(value: Optional[str], default: Vector = ZERO) -> Vector:
    values = _floats(value, 3, default)
    return values[0], values[1], values[2]


def _quaternion(value: Optional[str]) -> Quaternion:
    values = _floats(value, 4, IDENTITY_QUAT)
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 0.0:
        _fail("Quaternion must have non-zero norm")
    return tuple(item / norm for item in values)  # type: ignore[return-value]


def _format_float(value: float) -> str:
    if abs(value) < 5e-16:
        value = 0.0
    return f"{value:.15g}"


def _format_values(values: Iterable[float]) -> str:
    return " ".join(_format_float(value) for value in values)


def _vadd(left: Vector, right: Vector) -> Vector:
    return tuple(left[i] + right[i] for i in range(3))  # type: ignore[return-value]


def _vsub(left: Vector, right: Vector) -> Vector:
    return tuple(left[i] - right[i] for i in range(3))  # type: ignore[return-value]


def _vscale(value: Vector, scale: float) -> Vector:
    return tuple(item * scale for item in value)  # type: ignore[return-value]


def _dot(left: Vector, right: Vector) -> float:
    return sum(left[i] * right[i] for i in range(3))


def _qmul(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    value = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    return _quaternion(_format_values(value))


def _quat_matrix(quat: Quaternion) -> Matrix:
    w, x, y, z = quat
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _mat_vec(matrix: Matrix, value: Vector) -> Vector:
    return tuple(_dot(row, value) for row in matrix)  # type: ignore[return-value]


def _mat_mul(left: Matrix, right: Matrix) -> Matrix:
    right_t = _mat_transpose(right)
    return tuple(
        tuple(_dot(left[row], right_t[col]) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_transpose(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(matrix[col][row] for col in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def _mat_add(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(left[row][col] + right[row][col] for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _mat_scale(matrix: Matrix, scale: float) -> Matrix:
    return tuple(
        tuple(matrix[row][col] * scale for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _parallel_axis(offset: Vector, mass: float) -> Matrix:
    distance_squared = _dot(offset, offset)
    return tuple(
        tuple(
            mass
            * (
                (distance_squared if row == col else 0.0)
                - offset[row] * offset[col]
            )
            for col in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _compose(parent: Transform, child: Transform) -> Transform:
    parent_pos, parent_quat = parent
    child_pos, child_quat = child
    position = _vadd(parent_pos, _mat_vec(_quat_matrix(parent_quat), child_pos))
    return position, _qmul(parent_quat, child_quat)


def _element_transform(element: ET.Element) -> Transform:
    unsupported = ("axisangle", "euler", "xyaxes", "zaxis")
    present = [name for name in unsupported if element.get(name) is not None]
    if present:
        _fail(
            f"{element.tag} {element.get('name')!r} uses unsupported orientation "
            f"attribute(s): {', '.join(present)}"
        )
    return _vector(element.get("pos")), _quaternion(element.get("quat"))


def _full_inertia(element: ET.Element) -> Matrix:
    if element.get("quat") is not None:
        _fail("Composite inertials must express fullinertia in the body frame")
    values = _floats(element.get("fullinertia"), 6, ())
    ixx, iyy, izz, ixy, ixz, iyz = values
    return (
        (ixx, ixy, ixz),
        (ixy, iyy, iyz),
        (ixz, iyz, izz),
    )


def _matrix_to_full_inertia(matrix: Matrix) -> Tuple[float, ...]:
    return (
        matrix[0][0],
        matrix[1][1],
        matrix[2][2],
        matrix[0][1],
        matrix[0][2],
        matrix[1][2],
    )


def _mass_properties(body: ET.Element, transform: Transform) -> MassProperties:
    inertial = body.find("inertial")
    if inertial is None:
        _fail(f"Body {body.get('name')!r} has no explicit inertial")
    mass_text = inertial.get("mass")
    if mass_text is None:
        _fail(f"Body {body.get('name')!r} inertial has no mass")
    mass = float(mass_text)
    body_pos, body_quat = transform
    rotation = _quat_matrix(body_quat)
    com = _vadd(body_pos, _mat_vec(rotation, _vector(inertial.get("pos"))))
    inertia = _mat_mul(
        _mat_mul(rotation, _full_inertia(inertial)), _mat_transpose(rotation)
    )
    return MassProperties(mass=mass, com=com, inertia_at_com=inertia)


def _compound_mass_properties(parts: Sequence[MassProperties]) -> MassProperties:
    mass = sum(part.mass for part in parts)
    if mass <= 0.0:
        _fail("Composite mass must be positive")
    first_moment = ZERO
    for part in parts:
        first_moment = _vadd(first_moment, _vscale(part.com, part.mass))
    com = _vscale(first_moment, 1.0 / mass)
    inertia = _mat_scale(IDENTITY_MATRIX, 0.0)
    for part in parts:
        inertia = _mat_add(
            inertia,
            _mat_add(
                part.inertia_at_com,
                _parallel_axis(_vsub(part.com, com), part.mass),
            ),
        )
    return MassProperties(mass=mass, com=com, inertia_at_com=inertia)


def _body_parent_names(torso: ET.Element) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}

    def visit(body: ET.Element, parent_name: Optional[str]) -> None:
        name = body.get("name")
        if name is None:
            _fail("All ELF3 bodies must have names")
        result[name] = parent_name
        for child in body.findall("body"):
            visit(child, name)

    visit(torso, None)
    return result


def _validate_source(root: ET.Element) -> ET.Element:
    if root.tag != "mujoco" or root.get("model") != "elf3":
        _fail("Expected a <mujoco model='elf3'> source")

    worldbody = root.find("worldbody")
    if worldbody is None:
        _fail("Source has no worldbody")
    roots = worldbody.findall("body")
    if len(roots) != 1 or roots[0].get("name") != "torso_link":
        _fail("Source must have exactly one torso_link root body")
    torso = roots[0]

    body_names = tuple(body.get("name") for body in torso.iter("body"))
    expected_bodies = POLICY_BODY_NAMES + HEAD_BODY_NAMES
    if body_names != expected_bodies:
        _fail(
            "BXI body order changed.\n"
            f"Expected: {expected_bodies}\n"
            f"Actual:   {body_names}"
        )

    joint_names = tuple(
        joint.get("name") for joint in torso.iter("joint") if joint.get("name")
    )
    expected_joints = ("world_joint",) + POLICY_JOINT_NAMES + HEAD_JOINT_NAMES
    if joint_names != expected_joints:
        _fail(
            "BXI joint order changed.\n"
            f"Expected: {expected_joints}\n"
            f"Actual:   {joint_names}"
        )

    actuator = root.find("actuator")
    if actuator is None:
        _fail("Source has no actuator section")
    if any(child.tag != "motor" for child in actuator):
        _fail("Active BXI actuator section must contain motors only")
    motor_names = tuple(motor.get("joint") for motor in actuator.findall("motor"))
    expected_motors = POLICY_JOINT_NAMES + HEAD_JOINT_NAMES
    if motor_names != expected_motors:
        _fail(
            "BXI motor order changed.\n"
            f"Expected: {expected_motors}\n"
            f"Actual:   {motor_names}"
        )

    parents = _body_parent_names(torso)
    if parents["head_z_link"] != "torso_link":
        _fail("head_z_link must be a direct child of torso_link")
    if parents["head_y_link"] != "head_z_link":
        _fail("head_y_link must be a direct child of head_z_link")
    for name in HEAD_JOINT_NAMES:
        joint = torso.find(f".//joint[@name='{name}']")
        if joint is None or float(joint.get("ref", "0")) != 0.0:
            _fail(f"{name} must have zero reference angle before it can be fixed")

    for body in torso.iter("body"):
        if body.find("inertial") is None:
            _fail(f"Body {body.get('name')!r} has no explicit inertial")
    return torso


def _head_transforms(torso: ET.Element) -> Dict[str, Transform]:
    head_z = torso.find("body[@name='head_z_link']")
    if head_z is None:
        _fail("torso_link has no head_z_link child")
    result: Dict[str, Transform] = {}

    def visit(body: ET.Element, parent: Transform) -> None:
        name = body.get("name")
        if name is None:
            _fail("Head body is missing a name")
        transform = _compose(parent, _element_transform(body))
        result[name] = transform
        for child in body.findall("body"):
            visit(child, transform)

    visit(head_z, (ZERO, IDENTITY_QUAT))
    if tuple(result) != HEAD_BODY_NAMES:
        _fail(f"Unexpected fixed-head body hierarchy: {tuple(result)}")
    return result


def _transform_geom(geom: ET.Element, body_transform: Transform) -> ET.Element:
    geom = copy.deepcopy(geom)
    if geom.get("fromto") is not None:
        values = _floats(geom.get("fromto"), 6, ())
        start = values[0], values[1], values[2]
        end = values[3], values[4], values[5]
        start = _compose(body_transform, (start, IDENTITY_QUAT))[0]
        end = _compose(body_transform, (end, IDENTITY_QUAT))[0]
        geom.set("fromto", _format_values(start + end))
        geom.attrib.pop("pos", None)
        for name in ("quat", "axisangle", "euler", "xyaxes", "zaxis"):
            geom.attrib.pop(name, None)
        return geom

    transform = _compose(body_transform, _element_transform(geom))
    position, quaternion = transform
    if any(abs(value) > 5e-16 for value in position):
        geom.set("pos", _format_values(position))
    else:
        geom.attrib.pop("pos", None)
    if any(abs(quaternion[i] - IDENTITY_QUAT[i]) > 5e-15 for i in range(4)):
        geom.set("quat", _format_values(quaternion))
    else:
        geom.attrib.pop("quat", None)
    return geom


def _fixed_head_geoms(
    torso: ET.Element, transforms: Mapping[str, Transform]
) -> List[ET.Element]:
    result: List[ET.Element] = []
    for name in HEAD_BODY_NAMES:
        body = torso.find(f".//body[@name='{name}']")
        if body is None:
            _fail(f"Missing {name}")
        direct_geoms = body.findall("geom")
        if not direct_geoms:
            _fail(f"{name} has no geometry to fix to torso_link")
        result.extend(_transform_geom(geom, transforms[name]) for geom in direct_geoms)
    return result


def _strip_non_robot_elements(element: ET.Element) -> None:
    for child in list(element):
        if child.tag is ET.Comment or child.tag in {"camera", "light", "include"}:
            element.remove(child)
        else:
            _strip_non_robot_elements(child)


def _replace_torso_inertial(torso: ET.Element, properties: MassProperties) -> None:
    inertial = torso.find("inertial")
    if inertial is None:
        _fail("torso_link has no inertial")
    inertial.attrib.clear()
    inertial.set("pos", _format_values(properties.com))
    inertial.set("mass", _format_float(properties.mass))
    inertial.set(
        "fullinertia",
        _format_values(_matrix_to_full_inertia(properties.inertia_at_com)),
    )


def _insert_fixed_head_geoms(torso: ET.Element, geoms: Sequence[ET.Element]) -> None:
    children = list(torso)
    first_child_body = next(
        (index for index, child in enumerate(children) if child.tag == "body"),
        len(children),
    )
    torso.insert(first_child_body, ET.Comment(" head geoms fixed at head_z=head_y=0 "))
    for offset, geom in enumerate(geoms, start=1):
        torso.insert(first_child_body + offset, geom)


def _remove_head_bodies(torso: ET.Element) -> None:
    head_z = torso.find("body[@name='head_z_link']")
    if head_z is None:
        _fail("Could not remove head_z_link")
    torso.remove(head_z)


def _model_mass_properties(root: ET.Element) -> MassProperties:
    worldbody = root.find("worldbody")
    if worldbody is None:
        _fail("Model has no worldbody")
    parts: List[MassProperties] = []

    def visit(body: ET.Element, parent_transform: Transform) -> None:
        transform = _compose(parent_transform, _element_transform(body))
        parts.append(_mass_properties(body, transform))
        for child in body.findall("body"):
            visit(child, transform)

    for body in worldbody.findall("body"):
        visit(body, (ZERO, IDENTITY_QUAT))
    return _compound_mass_properties(parts)


def _mesh_source_dir(source: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    if compiler is None or compiler.get("meshdir") is None:
        _fail("Source compiler must define meshdir")
    return (source.parent / compiler.get("meshdir", "")).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_assets(
    asset: ET.Element, source_mesh_dir: Path, output_mesh_dir: Path
) -> Dict[Path, Path]:
    copy_plan: Dict[Path, Path] = {}
    mesh_names = tuple(mesh.get("name") for mesh in asset.findall("mesh"))
    # Asset declaration order follows the source XML (the head meshes are next
    # to the torso mesh), while body traversal order puts the head branch last.
    expected_mesh_names = (
        POLICY_BODY_NAMES[:1] + HEAD_BODY_NAMES + POLICY_BODY_NAMES[1:]
    )
    if mesh_names != expected_mesh_names:
        _fail(
            "BXI mesh order changed.\n"
            f"Expected: {expected_mesh_names}\n"
            f"Actual:   {mesh_names}"
        )

    for mesh in asset.findall("mesh"):
        source_name = mesh.get("file")
        if source_name is None:
            _fail(f"Mesh {mesh.get('name')!r} has no file")
        source_file = source_mesh_dir / source_name
        if not source_file.is_file():
            _fail(f"Missing source mesh: {source_file}")
        output_name = COPIED_MESH_FILES.get(source_name, source_name)
        output_file = output_mesh_dir / output_name
        mesh.set("file", output_name)
        mesh.attrib.pop("content_type", None)

        if source_name in COPIED_MESH_FILES:
            copy_plan[source_file] = output_file
        else:
            if not output_file.is_file():
                _fail(f"Missing shared ProtoMotions mesh: {output_file}")
            if _sha256(source_file) != _sha256(output_file):
                _fail(
                    f"Shared mesh {source_name} differs from BXI. Add an isolated "
                    "name to COPIED_MESH_FILES instead of overwriting legacy ELF3."
                )
    return copy_plan


def _clean_whitespace(element: ET.Element) -> None:
    if element.text is not None and not element.text.strip():
        element.text = None
    if element.tail is not None and not element.tail.strip():
        element.tail = None
    for child in element:
        _clean_whitespace(child)


def _indent(element: ET.Element, level: int = 0) -> None:
    """Backport the relevant behavior of ElementTree.indent for Python 3.8."""

    prefix = "\n" + "  " * level
    child_prefix = "\n" + "  " * (level + 1)
    children = list(element)
    if children:
        if element.text is None or not element.text.strip():
            element.text = child_prefix
        for index, child in enumerate(children):
            _indent(child, level + 1)
            if child.tail is None or not child.tail.strip():
                child.tail = child_prefix if index + 1 < len(children) else prefix


def _assert_close(label: str, left: float, right: float, tolerance: float) -> None:
    if abs(left - right) > tolerance:
        _fail(f"{label} mismatch: {left:.16g} vs {right:.16g}")


def build(source: Path, output: Path, output_mesh_dir: Path) -> BuildResult:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    source_tree = ET.parse(source, parser=parser)
    source_root = source_tree.getroot()
    source_torso = _validate_source(source_root)
    source_system = _model_mass_properties(source_root)

    head_transforms = _head_transforms(source_torso)
    torso_part = _mass_properties(source_torso, (ZERO, IDENTITY_QUAT))
    head_parts = [
        _mass_properties(
            source_torso.find(f".//body[@name='{name}']"),  # type: ignore[arg-type]
            head_transforms[name],
        )
        for name in HEAD_BODY_NAMES
    ]
    compound_torso = _compound_mass_properties([torso_part] + head_parts)
    fixed_geoms = _fixed_head_geoms(source_torso, head_transforms)

    output_root = ET.Element("mujoco", {"model": "elf3_bxi"})
    output_root.append(
        ET.Comment(
            " Generated by data/scripts/build_elf3_bxi_mjcf.py; do not edit manually. "
        )
    )
    compiler = copy.deepcopy(source_root.find("compiler"))
    if compiler is None:
        _fail("Source has no compiler section")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("meshdir", os.path.relpath(output_mesh_dir, output.parent))
    output_root.append(compiler)
    output_root.append(ET.Element("option", {"timestep": "0.002"}))

    defaults = source_root.find("default")
    assets = source_root.find("asset")
    if defaults is None or assets is None:
        _fail("Source must define default and asset sections")
    output_root.append(copy.deepcopy(defaults))
    output_assets = copy.deepcopy(assets)
    copy_plan = _prepare_assets(
        output_assets, _mesh_source_dir(source, source_root), output_mesh_dir
    )
    output_root.append(output_assets)

    output_worldbody = ET.SubElement(output_root, "worldbody")
    output_torso = copy.deepcopy(source_torso)
    _strip_non_robot_elements(output_torso)
    _remove_head_bodies(output_torso)
    _replace_torso_inertial(output_torso, compound_torso)
    _insert_fixed_head_geoms(output_torso, fixed_geoms)
    output_worldbody.append(output_torso)

    output_actuator = ET.SubElement(output_root, "actuator")
    source_motors = {
        motor.get("joint"): motor for motor in source_root.findall("./actuator/motor")
    }
    for joint_name in POLICY_JOINT_NAMES:
        motor = source_motors.get(joint_name)
        if motor is None:
            _fail(f"Missing policy motor {joint_name}")
        output_actuator.append(copy.deepcopy(motor))

    output_bodies = tuple(
        body.get("name") for body in output_torso.iter("body")
    )
    output_joints = tuple(
        joint.get("name")
        for joint in output_torso.iter("joint")
        if joint.get("name") != "world_joint"
    )
    output_motors = tuple(
        motor.get("joint") for motor in output_actuator.findall("motor")
    )
    if output_bodies != POLICY_BODY_NAMES:
        _fail(f"Output body contract mismatch: {output_bodies}")
    if output_joints != POLICY_JOINT_NAMES:
        _fail(f"Output joint contract mismatch: {output_joints}")
    if output_motors != POLICY_JOINT_NAMES:
        _fail(f"Output motor contract mismatch: {output_motors}")
    forbidden_tags = {"camera", "light", "sensor", "include"}
    leaked = [element.tag for element in output_root.iter() if element.tag in forbidden_tags]
    if leaked:
        _fail(f"Non-robot elements leaked into output: {leaked}")

    output_system = _model_mass_properties(output_root)
    _assert_close("total mass", source_system.mass, output_system.mass, 1e-10)
    for axis, source_value, output_value in zip(
        "xyz", source_system.com, output_system.com
    ):
        _assert_close(f"zero-pose COM {axis}", source_value, output_value, 1e-10)
    for row in range(3):
        for column in range(3):
            _assert_close(
                f"zero-pose inertia [{row}, {column}]",
                source_system.inertia_at_com[row][column],
                output_system.inertia_at_com[row][column],
                1e-10,
            )

    _clean_whitespace(output_root)
    _indent(output_root)
    xml_bytes = ET.tostring(
        output_root, encoding="utf-8", xml_declaration=True, short_empty_elements=True
    ) + b"\n"
    return BuildResult(
        xml_bytes=xml_bytes,
        copied_meshes=copy_plan,
        torso=compound_torso,
        source_mass=source_system.mass,
        source_com=source_system.com,
        source_inertia=source_system.inertia_at_com,
        output_mass=output_system.mass,
        output_com=output_system.com,
        output_inertia=output_system.inertia_at_com,
    )


def _write_if_changed(path: Path, content: bytes) -> bool:
    if path.is_file() and path.read_bytes() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return True


def _check_current(path: Path, content: bytes) -> None:
    if not path.is_file():
        _fail(f"Generated file is missing: {path}")
    if path.read_bytes() != content:
        _fail(f"Generated file is stale: {path}")


def _print_summary(result: BuildResult, output: Path, check: bool) -> None:
    action = "Verified" if check else "Generated"
    print(f"{action}: {output}")
    print("Contract: 30 bodies, 29 hinge joints, 29 motors, timestep=0.002 s")
    print(f"Composite torso mass: {_format_float(result.torso.mass)} kg")
    print(f"Composite torso COM:  {_format_values(result.torso.com)} m")
    print(
        "Composite torso fullinertia: "
        + _format_values(_matrix_to_full_inertia(result.torso.inertia_at_com))
        + " kg m^2"
    )
    print(f"Source total mass: {_format_float(result.source_mass)} kg")
    print(f"Output total mass: {_format_float(result.output_mass)} kg")
    print(f"Source zero-pose COM: {_format_values(result.source_com)} m")
    print(f"Output zero-pose COM: {_format_values(result.output_com)} m")
    print(
        "Source zero-pose fullinertia: "
        + _format_values(_matrix_to_full_inertia(result.source_inertia))
        + " kg m^2"
    )
    print(
        "Output zero-pose fullinertia: "
        + _format_values(_matrix_to_full_inertia(result.output_inertia))
        + " kg m^2"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mesh-output-dir", type=Path, default=DEFAULT_MESH_OUTPUT_DIR
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the generated XML and copied meshes are up to date.",
    )
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()
    output_mesh_dir = args.mesh_output_dir.resolve()

    try:
        result = build(source, output, output_mesh_dir)
        if args.check:
            _check_current(output, result.xml_bytes)
            for source_mesh, output_mesh in result.copied_meshes.items():
                _check_current(output_mesh, source_mesh.read_bytes())
        else:
            _write_if_changed(output, result.xml_bytes)
            for source_mesh, output_mesh in result.copied_meshes.items():
                _write_if_changed(output_mesh, source_mesh.read_bytes())
        _print_summary(result, output, args.check)
    except (OSError, ET.ParseError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
