# gnn_assembler/planning

Decides the disassembly order. The network only changes which part is tried first. A move is kept only when the checker accepts it.

| File | What it does |
| --- | --- |
| `physics.py` | Proposes extractions. Prismatic joints try the joint axis. Revolute joints try a 90° or 180° turn and then a lift, then a helical sweep. Other directions are tried after that. A candidate is feasible when the moving boxes do not penetrate the remaining parts, the part does not pass through the ground, and every remaining part still rests on the ground or on another part. |
| `search.py` | Breadth-first search over which part to remove. If the search visits too many states it falls back to taking the first feasible part at each step. `reverse_sequence` flips the order. The stored motion stays the extraction from the seated pose. Playing it from `t = 1` back to `t = 0` is the forward insertion. |
