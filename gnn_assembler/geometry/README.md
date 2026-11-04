# gnn_assembler/geometry

Turns part meshes into a contact graph the network and the planner can both read.

| File | What it does |
| --- | --- |
| `meshutil.py` | Builds a triangle mesh for a box, reads and writes `.obj` / `.stl` with Trimesh, and measures the distance between two meshes. |
| `subdivide.py` | Splits every triangle at its edge midpoints. `subdivide_batch` rewrites a dataset and copies `assembly.json` unchanged. Collision boxes are not subdivided. |
| `collision.py` | Oriented-box overlap, support on the ground or on another part, and the rigid motions used for extraction: translation, a simultaneous helix, and a turn-then-lift. |
| `graph.py` | Connects parts that touch. A pair that can translate apart is prismatic. A wrapped pair that cannot translate is revolute. Anything else is rigid. `assembly.json` overrides that guess. `featurize_state` builds the 17-number node vector: centroid, box size, volume, contact degree, a joint multi-hot, the mesh center of gravity, and the direction from the ground-facing face to that center. |
