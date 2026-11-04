"""Contact graphs, joint labels, and per-state GNN features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnn_assembler.assembly import Assembly, Part
from gnn_assembler.geometry.collision import (
    WORLD_DIRECTIONS,
    can_separate,
    contact_point,
    obb_gap,
    obb_z_limit,
    part_com,
    part_volume,
)
from gnn_assembler.geometry.meshutil import mesh_distance

FEATURE_DIM = 11
JOINT_TYPES = ("rigid", "revolute", "prismatic")
CONTACT_TOL = 1e-4


@dataclass
class ContactEdge:
    a: int
    b: int
    joint: str
    axis: np.ndarray
    point: np.ndarray


@dataclass
class ContactGraph:
    edges: list[ContactEdge]

    def pair(self, i: int, j: int) -> ContactEdge | None:
        for edge in self.edges:
            if {edge.a, edge.b} == {i, j}:
                return edge
        return None

    def incident(self, index: int) -> list[ContactEdge]:
        return [edge for edge in self.edges if edge.a == index or edge.b == index]


def _override_map(assembly: Assembly) -> dict[frozenset[str], tuple[str, np.ndarray]]:
    mapping: dict[frozenset[str], tuple[str, np.ndarray]] = {}
    for joint in assembly.joint_overrides:
        if joint.joint_type not in JOINT_TYPES:
            raise ValueError(f"Unknown joint type {joint.joint_type}")
        axis = joint.axis / max(float(np.linalg.norm(joint.axis)), 1e-12)
        mapping[frozenset((joint.a, joint.b))] = (joint.joint_type, axis)
    return mapping


def _surrounds(sleeve: Part, core: Part) -> bool:
    com = part_com(core)
    centers = np.stack([box.center for box in sleeve.obbs])
    dx = centers[:, 0] - com[0]
    dy = centers[:, 1] - com[1]
    core_z0 = min(obb_z_limit(box, -1.0) for box in core.obbs)
    core_z1 = max(obb_z_limit(box, 1.0) for box in core.obbs)
    z_overlap = any(
        obb_z_limit(box, -1.0) < core_z1 and obb_z_limit(box, 1.0) > core_z0 for box in sleeve.obbs
    )
    return bool(
        z_overlap
        and np.any(dx < -1e-3)
        and np.any(dx > 1e-3)
        and np.any(dy < -1e-3)
        and np.any(dy > 1e-3)
    )


def infer_joint(part_a: Part, part_b: Part) -> tuple[str, np.ndarray]:
    separating: set[int] = set()
    for direction in WORLD_DIRECTIONS:
        if can_separate(part_a, part_b, direction):
            separating.add(int(np.argmax(np.abs(direction))))
    if separating:
        # A pocket that can still lift apart is prismatic. Prefer the vertical mate when it is free.
        axis_index = 2 if 2 in separating else sorted(separating)[0]
        return "prismatic", np.eye(3)[axis_index]
    if _surrounds(part_a, part_b) or _surrounds(part_b, part_a):
        return "revolute", np.array([0.0, 0.0, 1.0])
    return "rigid", np.array([0.0, 0.0, 1.0])


def _parts_touch(part_a: Part, part_b: Part, tol: float) -> tuple[bool, np.ndarray]:
    best_gap = float("inf")
    point = np.zeros(3)
    for left in part_a.obbs:
        for right in part_b.obbs:
            gap = obb_gap(left, right)
            if gap < best_gap:
                best_gap = gap
                point = contact_point(left, right)
    mesh_gap = mesh_distance(part_a.vertices, part_a.faces, part_b.vertices, part_b.faces)
    gap = min(best_gap, mesh_gap)
    return gap <= tol, point


def build_contact_graph(
    assembly: Assembly,
    *,
    use_sidecar: bool = True,
    contact_tol: float = CONTACT_TOL,
) -> ContactGraph:
    overrides = _override_map(assembly) if use_sidecar else {}
    edges: list[ContactEdge] = []
    for i in range(len(assembly.parts)):
        for j in range(i + 1, len(assembly.parts)):
            touching, point = _parts_touch(assembly.parts[i], assembly.parts[j], contact_tol)
            if not touching:
                continue
            key = frozenset((assembly.parts[i].name, assembly.parts[j].name))
            if key in overrides:
                joint, axis = overrides[key]
            else:
                joint, axis = infer_joint(assembly.parts[i], assembly.parts[j])
            edges.append(ContactEdge(i, j, joint, axis, point))
    return ContactGraph(edges)


def assembly_frame(assembly: Assembly) -> tuple[np.ndarray, float]:
    points = [part.vertices for part in assembly.parts if len(part.vertices)]
    if not points:
        stacked = np.vstack([part_com(part) for part in assembly.parts])
    else:
        stacked = np.vstack(points)
    low = stacked.min(axis=0)
    high = stacked.max(axis=0)
    scale = max(float(np.linalg.norm(high - low)), 1e-6)
    return low, scale


def _bbox(part: Part) -> tuple[np.ndarray, np.ndarray]:
    low = part.vertices.min(axis=0)
    high = part.vertices.max(axis=0)
    return low, high


def featurize_state(
    assembly: Assembly,
    graph: ContactGraph,
    remaining: set[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Node features and adjacency for the parts still assembled.

    Features: centroid (3), bbox size (3), volume (1), degree (1), joint multi-hot (3).
    Positions are scaled by the full assembly diagonal.
    """

    indices = sorted(remaining, key=lambda index: assembly.parts[index].name)
    origin, scale = assembly_frame(assembly)
    count = len(indices)
    features = np.zeros((count, FEATURE_DIM), dtype=np.float32)
    adjacency = np.zeros((count, count), dtype=np.float32)
    position = {part_index: row for row, part_index in enumerate(indices)}
    index_set = set(indices)
    degree = {part_index: 0 for part_index in indices}
    joints = {part_index: set() for part_index in indices}
    for edge in graph.edges:
        if edge.a not in index_set or edge.b not in index_set:
            continue
        adjacency[position[edge.a], position[edge.b]] = 1.0
        adjacency[position[edge.b], position[edge.a]] = 1.0
        degree[edge.a] += 1
        degree[edge.b] += 1
        joints[edge.a].add(edge.joint)
        joints[edge.b].add(edge.joint)
    normalizer = max(count - 1, 1)
    for row, part_index in enumerate(indices):
        part = assembly.parts[part_index]
        low, high = _bbox(part)
        centroid = part_com(part)
        features[row, 0:3] = (centroid - origin) / scale
        features[row, 3:6] = (high - low) / scale
        features[row, 6] = part_volume(part) / (scale**3)
        features[row, 7] = degree[part_index] / normalizer
        for joint_index, name in enumerate(JOINT_TYPES):
            if name in joints[part_index]:
                features[row, 8 + joint_index] = 1.0
    return features, adjacency, indices
