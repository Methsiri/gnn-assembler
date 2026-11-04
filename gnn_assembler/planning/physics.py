"""Candidate extraction motions and physical feasibility."""

from __future__ import annotations

import math

import numpy as np

from gnn_assembler.assembly import Assembly, Part
from gnn_assembler.geometry.collision import (
    WORLD_DIRECTIONS,
    Motion,
    clearance_distance,
    is_stable,
    obb_corners,
    part_com,
    sweep_clear,
)
from gnn_assembler.geometry.graph import ContactGraph


def _anchor(part: Part) -> np.ndarray:
    corners = np.vstack([obb_corners(box) for box in part.obbs])
    return (corners.min(axis=0) + corners.max(axis=0)) * 0.5


def propose_motions(
    assembly: Assembly,
    graph: ContactGraph,
    remaining: set[int],
    part_index: int,
) -> list[Motion]:
    part = assembly.parts[part_index]
    others = [assembly.parts[index] for index in remaining if index != part_index]
    motions: list[Motion] = []
    seen: set[tuple] = set()

    def add(motion: Motion) -> None:
        key = (
            motion.kind,
            tuple(np.round(motion.direction, 5)),
            round(motion.angle, 5),
            round(motion.distance, 5),
        )
        if key in seen:
            return
        seen.add(key)
        motions.append(motion)

    revolute_axes: list[tuple[np.ndarray, np.ndarray]] = []
    prismatic_axes: list[np.ndarray] = []
    for edge in graph.incident(part_index):
        other_index = edge.b if edge.a == part_index else edge.a
        if other_index not in remaining:
            continue
        axis = edge.axis / max(float(np.linalg.norm(edge.axis)), 1e-12)
        anchor = _anchor(assembly.parts[other_index])
        if edge.joint == "revolute":
            revolute_axes.append((axis, anchor))
        elif edge.joint == "prismatic":
            prismatic_axes.append(axis)

    angles = (-math.pi / 2.0, math.pi / 2.0, math.pi, -math.pi)
    for axis, anchor in revolute_axes:
        for sign in (1.0, -1.0):
            direction = axis * sign
            distance = clearance_distance(part, others, direction)
            for angle in angles:
                add(Motion("turn_then_lift", direction, distance, angle, anchor, axis))
                add(Motion("helix", direction, distance, angle, anchor, axis))
    for axis in prismatic_axes:
        for sign in (1.0, -1.0):
            direction = axis * sign
            add(
                Motion(
                    "translate",
                    direction,
                    clearance_distance(part, others, direction),
                    0.0,
                    part_com(part),
                    direction,
                )
            )
    for direction in WORLD_DIRECTIONS:
        add(
            Motion(
                "translate",
                direction,
                clearance_distance(part, others, direction),
                0.0,
                part_com(part),
                direction,
            )
        )
    return motions


def motion_is_feasible(
    assembly: Assembly,
    remaining: set[int],
    part_index: int,
    motion: Motion,
) -> bool:
    others = [assembly.parts[index] for index in sorted(remaining) if index != part_index]
    static = [box for part in others for box in part.obbs]
    if not sweep_clear(assembly.parts[part_index].obbs, static, motion, assembly.ground_z):
        return False
    return is_stable(others, assembly.ground_z)


def find_removal_motion(
    assembly: Assembly,
    graph: ContactGraph,
    remaining: set[int],
    part_index: int,
) -> Motion | None:
    for motion in propose_motions(assembly, graph, remaining, part_index):
        if motion_is_feasible(assembly, remaining, part_index, motion):
            return motion
    return None
