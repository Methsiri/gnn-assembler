"""Three small assemblies used to exercise the pipeline without external data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnn_assembler.geometry.meshutil import mesh_from_boxes, save_mesh

Bounds = list[tuple[list[float], list[float]]]


def write_samples(root: str | Path) -> None:
    root_path = Path(root)
    _write_assembly(
        root_path / "peg_hole",
        "peg_hole",
        parts={
            "base": _peg_base(),
            "peg": [([1.75, 1.75, 0.5], [2.25, 2.25, 2.0])],
        },
        masses={"base": 2.0, "peg": 0.3},
        joints=[
            {"a": "base", "b": "peg", "type": "prismatic", "axis": [0.0, 0.0, 1.0]},
        ],
    )
    _write_assembly(
        root_path / "rotational",
        "rotational",
        parts={
            "shaft": _shaft_boxes(),
            "collar": _collar_boxes(),
        },
        masses={"shaft": 1.5, "collar": 0.4},
        joints=[],
    )
    _write_assembly(
        root_path / "multi_stack",
        "stack",
        parts={
            "base": [([0.0, 0.0, 0.0], [4.0, 4.0, 1.0])],
            "mid": [([1.0, 1.0, 1.0], [3.0, 3.0, 2.0])],
            "top": [([1.5, 1.5, 2.0], [2.5, 2.5, 3.0])],
        },
        masses={"base": 2.0, "mid": 1.0, "top": 0.4},
        joints=[
            {"a": "base", "b": "mid", "type": "prismatic", "axis": [0.0, 0.0, 1.0]},
            {"a": "mid", "b": "top", "type": "prismatic", "axis": [0.0, 0.0, 1.0]},
        ],
    )


def _peg_base() -> Bounds:
    return [
        ([0.0, 0.0, 0.0], [4.0, 4.0, 0.5]),
        ([1.0, 1.0, 0.5], [1.5, 3.0, 1.5]),
        ([2.5, 1.0, 0.5], [3.0, 3.0, 1.5]),
        ([1.5, 1.0, 0.5], [2.5, 1.5, 1.5]),
        ([1.5, 2.5, 0.5], [2.5, 3.0, 1.5]),
    ]


def _shaft_boxes() -> Bounds:
    return [
        ([1.6, 1.6, 0.0], [2.4, 2.4, 4.0]),
        ([0.5, 0.5, 0.0], [3.5, 3.5, 0.4]),
        ([2.85, 1.88, 1.55], [3.35, 2.12, 1.90]),
    ]


def _collar_boxes() -> Bounds:
    return [
        ([2.9, 0.7, 0.4], [3.3, 3.3, 1.55]),
        ([0.7, 0.7, 0.4], [1.1, 3.3, 1.55]),
        ([0.7, 0.7, 0.4], [3.3, 1.1, 1.55]),
        ([0.7, 2.9, 0.4], [1.8, 3.3, 1.55]),
        ([2.2, 2.9, 0.4], [3.3, 3.3, 1.55]),
    ]


def _write_assembly(
    directory: Path,
    assembly_id: str,
    parts: dict[str, Bounds],
    masses: dict[str, float],
    joints: list[dict],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload_parts = []
    for name, bounds in parts.items():
        arrays = [(np.array(low, dtype=float), np.array(high, dtype=float)) for low, high in bounds]
        vertices, faces = mesh_from_boxes(arrays)
        filename = f"{name}.obj"
        save_mesh(directory / filename, vertices, faces)
        payload_parts.append(
            {
                "name": name,
                "file": filename,
                "mass": masses.get(name, 1.0),
                "boxes": [{"min": low, "max": high} for low, high in bounds],
            }
        )
    document = {
        "id": assembly_id,
        "ground_z": 0.0,
        "parts": payload_parts,
        "joints": joints,
    }
    (directory / "assembly.json").write_text(json.dumps(document, indent=2) + "\n")
