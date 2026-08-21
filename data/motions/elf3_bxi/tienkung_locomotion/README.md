# TienKung ELF3 BXI locomotion motions

This directory contains all 11 TienKung-Lab ELF3 source clips converted to the
ProtoMotions `.motion` format with the `elf3_bxi.xml` kinematic model:

- `amp/`: `run_walk`, `stand`, `stand_back`, `stand_run`, `walk_around`,
  `walk_left`, `walk_right`, and `walk_run`.
- `run/`: `run`.
- `walk/`: `walk`.
- `walk_old/`: legacy `walk`.

## Source and regeneration

The source pickles are under:

```text
/home/faw/workspace/bxi/TienKung-Lab-bxi/legged_lab/envs/elf3/datasets
```

Regenerate all 11 motions from the ProtoMotions repository root with:

```bash
/home/faw/workspace/IsaacLab/.venv/bin/python \
  data/scripts/convert_tienkung_elf3_pkl_to_proto.py \
  --input-path /home/faw/workspace/bxi/TienKung-Lab-bxi/legged_lab/envs/elf3/datasets \
  --output-dir data/motions/elf3_bxi/tienkung_locomotion \
  --force-remake
```

Without `--force-remake`, existing output files are safely skipped.

## Conversion contract

- Source `root_rot` quaternions are `xyzw`; the converter normalizes them and
  changes them to ProtoMotions' FK input order, `wxyz`.
- The source `dof_pos` columns are the fixed 29-joint ELF3 order. The converter
  verifies that this order matches `elf3_bxi.xml` before FK.
- Each source pickle's embedded floating-point `fps` value is retained; it is
  not rounded to 30 Hz.
- Source root positions are preserved exactly. The TienKung visualizer's
  `root_z += 0.3` display-only offset is not applied.

## Stage-1 steering set

[`../../../yaml_files/elf3_bxi_tienkung_steering_walk.yaml`](../../../yaml_files/elf3_bxi_tienkung_steering_walk.yaml)
uses five clips with fixed semantic sampling weights:

| Motion | Weight | Role |
| --- | ---: | --- |
| `walk/walk.motion` | 0.50 | Primary forward walking distribution |
| `amp/stand.motion` | 0.10 | Stop and near-zero-speed behavior |
| `amp/walk_around.motion` | 0.15 | Sustained curved walking |
| `amp/walk_left.motion` | 0.125 | Left-turn coverage |
| `amp/walk_right.motion` | 0.125 | Right-turn coverage |

The weights sum to 1.0 and are intentionally semantic rather than proportional
to clip duration.

The following converted clips remain available for later curriculum stages but
are intentionally excluded from Stage 1:

- `amp/stand_back.motion`: this clip moves backward relative to the robot's
  facing direction, while Stage 1 uses `enable_rand_facing=False` and therefore
  asks the robot to face its travel direction. Introduce it only with an
  independent-facing/backward-command curriculum.
- `run/run.motion`: running expands the speed and impact regime before the
  walking controller is stable.
- `amp/run_walk.motion`, `amp/stand_run.motion`, and `amp/walk_run.motion`:
  these transitions mix gait and speed regimes, so they are deferred until the
  corresponding walk/run command ranges are enabled.
- `walk_old/walk.motion`: this is a legacy, redundant walking clip; excluding it
  avoids double-weighting an older version of the primary gait.
