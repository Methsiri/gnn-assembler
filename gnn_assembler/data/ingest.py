"""Load assembled part meshes from a directory."""

from __future__ import annotations

import json
from pathlib import Path

from gnn_assembler.assembly import Assembly, JointOverride, Part
from gnn_assembler.geometry.collision import obb_from_bounds
from gnn_assembler.geometry.meshutil import load_mesh
from gnn_assembler.geometry.subdivide import mesh_files


def load_assembly(directory: str | Path) -> Assembly:
    """Load one assembly whose parts are already in the assembled frame."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    meta_path = root / "assembly.json"
    if meta_path.exists():
        return _load_with_meta(root, json.loads(meta_path.read_text()))
    return _load_bare(root)


def discover(root: str | Path) -> list[Assembly]:
    """Load a single assembly directory or each child directory of a dataset."""

    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(path)
    if (path / "assembly.json").exists() or mesh_files(path):
        return [load_assembly(path)]
    assemblies = [
        load_assembly(child)
        for child in sorted(item for item in path.iterdir() if item.is_dir())
        if (child / "assembly.json").exists() or mesh_files(child)
    ]
    if not assemblies:
        raise FileNotFoundError(f"No assemblies found under {path}")
    return assemblies


def find_assembly(root: str | Path, assembly_id: str) -> Assembly:
    matches = [item for item in discover(root) if item.assembly_id == assembly_id]
    if not matches:
        known = ", ".join(item.assembly_id for item in discover(root))
        raise FileNotFoundError(f"Assembly {assembly_id!r} not found. Known ids: {known}")
    if len(matches) > 1:
        raise ValueError(f"More than one assembly is named {assembly_id!r}")
    return matches[0]


def _load_with_meta(root: Path, meta: dict) -> Assembly:
    parts: list[Part] = []
    for spec in meta.get("parts", []):
        mesh_path = root / spec["file"]
        vertices, faces = load_mesh(mesh_path)
        if spec.get("boxes"):
            obbs = [obb_from_bounds(box["min"], box["max"]) for box in spec["boxes"]]
        else:
            obbs = [obb_from_bounds(vertices.min(axis=0), vertices.max(axis=0))]
        parts.append(
            Part(
                name=spec.get("name", mesh_path.stem),
                vertices=vertices,
                faces=faces,
                obbs=obbs,
                mass=float(spec.get("mass", 1.0)),
            )
        )
    if not parts:
        raise ValueError(f"{root / 'assembly.json'} does not list any parts")
    joints = [
        JointOverride(
            a=spec["a"],
            b=spec["b"],
            joint_type=spec["type"],
            axis=spec.get("axis", [0.0, 0.0, 1.0]),
        )
        for spec in meta.get("joints", [])
    ]
    return Assembly(
        assembly_id=str(meta.get("id", root.name)),
        parts=parts,
        joint_overrides=joints,
        ground_z=float(meta.get("ground_z", 0.0)),
        source_dir=root,
    )


def _load_bare(root: Path) -> Assembly:
    files = mesh_files(root)
    if not files:
        raise FileNotFoundError(f"No meshes in {root}")
    parts: list[Part] = []
    for mesh_path in files:
        vertices, faces = load_mesh(mesh_path)
        obbs = [obb_from_bounds(vertices.min(axis=0), vertices.max(axis=0))]
        parts.append(Part(mesh_path.stem, vertices, faces, obbs, 1.0))
    return Assembly(root.name, parts, [], 0.0, root)
