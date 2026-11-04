# tests

Pytest checks for the pipeline. They build the sample assemblies in a temporary directory, so they do not need `joint_assembly` or a GPU.

| File | What it checks |
| --- | --- |
| `test_geometry.py` | Subdivision increases the vertex count and keeps `assembly.json`. The stack, peg, and collar graphs get the expected contacts and joint types. |
| `test_planning.py` | Removal order for the stack and the peg, a non-zero turn on the collar, and a round trip that brings every vertex back to the designed pose. SCD is near 0 and CA is 1. |
| `test_model.py` | The network emits one score per part. A two-epoch training run writes a checkpoint that still plans the stack. |
| `test_job.py` | Task prerequisites, unique G-code names, and a `.3mf` zip that contains the model, the OPC relationship files, `Tasks.xml`, and the G-code. Also runs the `plan` command. |

```bash
python -m pytest
```
