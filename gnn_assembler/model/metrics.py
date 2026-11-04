"""Shape Chamfer Distance and Connectivity Accuracy."""

from __future__ import annotations

import numpy as np

from gnn_assembler.assembly import Assembly
from gnn_assembler.geometry.collision import inverse_transform_points, transform_points
from gnn_assembler.geometry.graph import ContactGraph, assembly_frame
from gnn_assembler.planning.search import RemovalStep


def shape_chamfer_distance(source: np.ndarray, target: np.ndarray) -> float:
    """Bidirectional mean nearest-vertex distance."""

    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return 0.0
    deltas = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    return float(deltas.min(axis=1).mean() + deltas.min(axis=0).mean())


def connectivity_accuracy(pairs: list[tuple[np.ndarray, np.ndarray]], scale: float) -> float:
    """Fraction of contact pairs whose endpoints stay within ``0.01 * scale``."""

    tau = 0.01 * max(scale, 1e-6)
    if not pairs:
        return 1.0
    matched = sum(float(np.linalg.norm(left - right)) <= tau for left, right in pairs)
    return matched / len(pairs)


def evaluate_sequence(
    assembly: Assembly,
    graph: ContactGraph,
    disassembly: list[RemovalStep],
) -> tuple[float, float]:
    """Score the seated assembly produced by reversing each extraction."""

    by_part = {step.part_index: step.motion for step in disassembly}
    clouds = []
    for index, part in enumerate(assembly.parts):
        vertices = part.vertices
        motion = by_part.get(index)
        if motion is not None:
            extracted = transform_points(vertices, motion, 1.0)
            vertices = inverse_transform_points(extracted, motion, 1.0)
        clouds.append(vertices)
    design = np.vstack([part.vertices for part in assembly.parts])
    predicted = np.vstack(clouds)
    _origin, scale = assembly_frame(assembly)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for edge in graph.edges:
        left = edge.point.reshape(1, 3)
        right = edge.point.reshape(1, 3)
        if edge.a in by_part:
            left = inverse_transform_points(transform_points(left, by_part[edge.a], 1.0), by_part[edge.a], 1.0)
        if edge.b in by_part:
            right = inverse_transform_points(transform_points(right, by_part[edge.b], 1.0), by_part[edge.b], 1.0)
        pairs.append((left.reshape(3), right.reshape(3)))
    return shape_chamfer_distance(design, predicted), connectivity_accuracy(pairs, scale)
