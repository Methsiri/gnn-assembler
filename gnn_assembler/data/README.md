# gnn_assembler/data

Loads assemblies from disk and writes the three tiny sample assemblies.

| File | What it does |
| --- | --- |
| `ingest.py` | `discover` reads a dataset directory. One child folder is one assembly. A folder of `.obj` / `.stl` files is used as-is. `assembly.json` can add mass, collision boxes, and joint overrides. The assembly id is the `id` field, or the folder name when there is no JSON. |
| `samples.py` | `write_samples` builds `samples/peg_hole`, `samples/rotational`, and `samples/multi_stack`. |

Your own meshes do not go in this folder. Unzip datasets such as `joint_assembly.zip` under the repository `data/` directory. See `data/README.md` at the repository root.
