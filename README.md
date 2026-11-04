# gnn-assembler

Plan a physically feasible assembly sequence from part meshes and pack it as a [3MF](https://3mf.io/) job. A small graph network ranks which part to remove next. A geometric checker accepts or rejects that extraction. The confirmed disassembly is time-reversed into the forward assembly, then written as `Tasks.xml`, one G-code file per step, and a 3D model bound together by Open Packaging Conventions relationships.

The pipeline follows the same assembly-by-disassembly idea as physics-based sequence planning: subdivide meshes, build a contact graph, learn a removal heuristic, search with that heuristic, then reverse the path. Shape Chamfer Distance (SCD) and Connectivity Accuracy (CA) score the seated result. It does not download external datasets, provision a cluster, or reimplement a differentiable rigid-body engine.

## Stages

1. **Ingest.** `gnn-assemble generate-samples` writes three assemblies: `samples/peg_hole` (a peg in a pocket), `samples/rotational` (a bayonet collar on a shaft), and `samples/multi_stack` (id `stack`, three blocks). Any directory of world-frame `.obj` or `.stl` files is also an assembly. Optional `assembly.json` records mass, oriented-box collision primitives, and joint overrides (`rigid`, `revolute`, `prismatic`).
2. **Subdivide and build the contact graph.** `gnn-assemble subdivide` midpoint-splits every triangle so contact sampling is denser. Parts become graph nodes. An edge is a near or touching pair. Joint type comes from `assembly.json` when present, otherwise from geometry: a free separating translation is prismatic, a wrapped pair that cannot translate is revolute, and a locked pair is rigid.
3. **Train the removal heuristic.** `gnn-assemble train` labels the next removable part with the physics search and fits a two-layer message-passing network (binary cross-entropy). CUDA is used when it is available. The checkpoint is optional at plan time; without it, higher parts are tried first.
4. **Search and reverse.** Breadth-first search tries parts in heuristic order. A move is kept only when the part sweeps out without penetrating the rest and the remainder stays supported. Revolute joints try a rotation about the joint axis followed by a lift, then a simultaneous helical sweep. The disassembly is reversed to build the part: later extractions are inserted first, with direction and angle negated.
5. **Trajectories and G-code.** Each insertion is an approach, an align, and a seat. A fixed linear map adds a small residual to the approach point. Every step becomes `Attachments/<uuid>.gcode` (`G21`, `G90`, one rapid, then linear moves).
6. **Task DAG.** `Attachments/Tasks.xml` has one `<Task>` per part: seated `<Position>`, `<Rotation>`, `<Insertion>`, `<GCode>`, and `<Prerequisites>` listing earlier tasks it contacts.
7. **3MF package.** The archive is built in memory with deflated ZIP entries: `3D/3dmodel.model`, `[Content_Types].xml`, `_rels/.rels`, and `3D/_rels/3dmodel.model.rels` linking the task XML and G-code to the model.

## Install

```bash
pip install -e ".[dev]"
```

`environment.yml` is the same set of dependencies for a Conda environment (Python 3.11, NumPy, PyTorch, Trimesh, pytest).

## Commands

```bash
gnn-assemble generate-samples --out samples
gnn-assemble subdivide --source samples/multi_stack --target /tmp/stack_dense --levels 1
gnn-assemble train --data samples --epochs 20 --out checkpoints/heuristic.pt
gnn-assemble plan --dir samples --id stack --output stack.3mf
gnn-assemble plan --dir samples --id rotational --checkpoint checkpoints/heuristic.pt --output rotational.3mf
```

`plan` prints the disassembly, the forward assembly, SCD, and CA. A feasible reverse of a checked plan sits on the designed geometry, so SCD is near zero and CA is 1 on these samples. CA uses a threshold of `0.01` times the assembly diagonal.

## External meshes

The two-part, rotational, and multi-part research sets often cited with this problem are about 221.1MB, 5.4MB, and 778.4MB. A large part of that data is derived from the Fusion 360 Gallery dataset and stays under that license. This project does not download it. Point `--dir` at a local folder whose children are assemblies of already-posed meshes. Add `assembly.json` boxes when a part is concave; otherwise each mesh is treated as one bounding box.

## Tests

```bash
python -m pytest
```
