# data

Local datasets. This folder is for meshes you train and plan on. The code that reads them lives in `gnn_assembler/data`. Nothing in this folder is required for the built-in samples.

## joint_assembly.zip

Unzip the two-part research set here so the numbered assembly folders are direct children of `data/joint_assembly/`:

```text
data/
  joint_assembly/
    00000/
      0.obj
      1.obj
    00001/
      ...
```

```bash
unzip joint_assembly.zip -d data
gnn-assemble train --data data/joint_assembly --epochs 20 --out checkpoints/heuristic.pt
gnn-assemble plan --dir data/joint_assembly --id 00000 --output job.3mf
```

If the archive has an extra wrapper directory, pass `--data` the folder that actually contains `00000`, `00001`, and the rest.

Each child folder is one assembly. Parts must already be in the assembled pose. A mesh with no `assembly.json` is treated as a single bounding box, which is a poor collision shape for interlocking parts. For a first run, copy a few id folders into `data/joint_assembly_small` and train on that. Training plans every assembly in the folder before it updates the network.

The zip and the extracted folders are gitignored. The set is large, and much of it is derived from the Fusion 360 Gallery dataset under that dataset’s license.
