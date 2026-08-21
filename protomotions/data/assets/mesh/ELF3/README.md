# BXI ELF3 Description Assets

This directory contains mesh assets for the BXI ELF3 humanoid. They were
adapted from the `elf3_lite` asset distributed by the TienKung-Lab project.
The robot-only MJCFs are:

- `protomotions/data/assets/mjcf/elf3.xml`, the legacy ELF3-lite model.
- `protomotions/data/assets/mjcf/elf3_bxi.xml`, the 29-DOF model aligned with
  the BXI controller simulation.

`head_y_link.STL`, `head_z_link.STL`, `torso_link_bxi.STL`, and the two
`*_ankle_x_link_bxi.STL` files are used by `elf3_bxi.xml`. They are synchronized
from the BXI controller's `data/mujoco_simulation/meshes` directory by
`data/scripts/build_elf3_bxi_mjcf.py`. The generator verifies that every other
BXI mesh is byte-identical to the shared legacy file before reusing it, so the
legacy `elf3.xml` assets are never overwritten.

Source: [MelodyAI/TienKung-Lab-bxi](https://github.com/MelodyAI/TienKung-Lab-bxi)
at commit
`e6b51aeaca41d53817586b1a75239c6cdee67292`.

The source asset and these redistributed files are licensed under the
[BSD-3-Clause License](LICENSE).
