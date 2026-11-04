"""Assembly data shared by ingest, planning, and packaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OBB:
    """Oriented box. ``axes`` columns are the local unit axes in world frame."""

    center: np.ndarray
    half: np.ndarray
    axes: np.ndarray

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.half = np.asarray(self.half, dtype=float).reshape(3)
        self.axes = np.asarray(self.axes, dtype=float).reshape(3, 3)

    def copy(self) -> OBB:
        return OBB(self.center.copy(), self.half.copy(), self.axes.copy())


@dataclass
class Part:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    obbs: list[OBB]
    mass: float = 1.0

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        self.mass = float(self.mass)


@dataclass
class JointOverride:
    a: str
    b: str
    joint_type: str
    axis: np.ndarray

    def __post_init__(self) -> None:
        self.axis = np.asarray(self.axis, dtype=float).reshape(3)


@dataclass
class Assembly:
    assembly_id: str
    parts: list[Part]
    joint_overrides: list[JointOverride] = field(default_factory=list)
    ground_z: float = 0.0
    source_dir: Path | None = None

    def part_index(self, name: str) -> int:
        for index, part in enumerate(self.parts):
            if part.name == name:
                return index
        raise KeyError(name)

    def renamed(self, parts: list[Part]) -> Assembly:
        return Assembly(
            assembly_id=self.assembly_id,
            parts=parts,
            joint_overrides=list(self.joint_overrides),
            ground_z=self.ground_z,
            source_dir=self.source_dir,
        )
