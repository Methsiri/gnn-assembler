# gnn_assembler

Python package for the whole pipeline. `python -m gnn_assembler` and the `gnn-assemble` command both enter through `cli.py`.

| File | What it does |
| --- | --- |
| `cli.py` | Commands: `generate-samples`, `subdivide`, `train`, `plan`. |
| `pipeline.py` | One planning run: subdivide, search, write tasks, pack a `.3mf`. |
| `assembly.py` | Shared types: a part mesh, its collision boxes, and an assembly. |

The stages live in the child folders:

- `data` loads meshes and writes the built-in samples.
- `geometry` subdivides meshes and builds the contact graph.
- `model` is the removal network, its training, and SCD / CA scores.
- `planning` checks each extraction and searches for an order.
- `execution` turns a reversed path into waypoints and G-code.
- `job` writes `Tasks.xml` and the 3MF archive.
