# gnn_assembler/execution

Hardware-facing path for one insertion. This is a fixed residual on the approach point, not a trained contact policy.

| File | What it does |
| --- | --- |
| `trajectory.py` | Samples the reversed extraction at the free pose, an intermediate pose, and the seated pose. A fixed linear map adds a small sideways offset to the approach point only, so the part still finishes on its designed seat. |
| `gcode.py` | Writes one program per step: millimeter units, absolute coordinates, a rapid to the approach, linear moves through the rest, then `M2`. The file name is a UUID, `Attachments/<uuid>.gcode`, inside the 3MF package. |
