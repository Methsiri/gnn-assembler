# gnn_assembler/model

The learned hint for which part to remove next, plus the scores reported after a plan.

| File | What it does |
| --- | --- |
| `gnn.py` | `RemovalGNN`: encode each part, two rounds of neighbor-mean message passing, then one score per part. CUDA is used when a device is not requested and a GPU is present. |
| `train.py` | Plans every training assembly with physics, labels the part that was actually removed, and fits the network with binary cross-entropy. `gnn-assemble train` saves `checkpoints/heuristic.pt`. That folder is created on demand and is not committed. |
| `metrics.py` | Shape Chamfer Distance is the bidirectional mean nearest-vertex distance between the designed mesh and the mesh after extract-and-restore. Connectivity Accuracy is the fraction of contact pairs that stay within `0.01` times the assembly diagonal. |

`plan` still runs with no checkpoint. In that case taller parts are tried first, and physics still accepts or rejects the move.
