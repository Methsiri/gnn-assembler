"""Oriented-box collision, rigid motions, and separation checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gnn_assembler.assembly import OBB, Part

OVERLAP_EPS = 1e-4
SUPPORT_EPS = 1e-3

WORLD_DIRECTIONS = (
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
)


@dataclass
class Motion:
    """Rigid extraction from the assembled pose (t=0) to the free pose (t=1)."""

    kind: str
    direction: np.ndarray
    distance: float
    angle: float
    axis_point: np.ndarray
    axis_dir: np.ndarray

    def __post_init__(self) -> None:
        self.direction = _unit(np.asarray(self.direction, dtype=float))
        self.axis_dir = _unit(np.asarray(self.axis_dir, dtype=float))
        self.axis_point = np.asarray(self.axis_point, dtype=float).reshape(3)
        self.distance = float(self.distance)
        self.angle = float(self.angle)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return vector / norm


def obb_from_bounds(low: np.ndarray, high: np.ndarray) -> OBB:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return OBB((low + high) * 0.5, (high - low) * 0.5, np.eye(3))


def obb_volume(box: OBB) -> float:
    return float(8.0 * box.half[0] * box.half[1] * box.half[2])


def part_volume(part: Part) -> float:
    return float(sum(obb_volume(box) for box in part.obbs))


def part_com(part: Part) -> np.ndarray:
    total = 0.0
    acc = np.zeros(3)
    for box in part.obbs:
        volume = obb_volume(box)
        acc += volume * box.center
        total += volume
    if total <= 0.0:
        if len(part.vertices) == 0:
            return np.zeros(3)
        return part.vertices.mean(axis=0)
    return acc / total


def obb_corners(box: OBB) -> np.ndarray:
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    return box.center + (signs * box.half) @ box.axes.T


def obb_z_limit(box: OBB, sign: float) -> float:
    return float(box.center[2] + sign * np.sum(box.half * np.abs(box.axes[2, :])))


def part_min_z(part: Part) -> float:
    return min(obb_z_limit(box, -1.0) for box in part.obbs)


def part_max_z(part: Part) -> float:
    return max(obb_z_limit(box, 1.0) for box in part.obbs)


def world_aabb(box: OBB) -> tuple[np.ndarray, np.ndarray]:
    corners = obb_corners(box)
    return corners.min(axis=0), corners.max(axis=0)


def aabb_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
    delta = np.zeros(3)
    for axis in range(3):
        if a1[axis] < b0[axis]:
            delta[axis] = b0[axis] - a1[axis]
        elif b1[axis] < a0[axis]:
            delta[axis] = a0[axis] - b1[axis]
    return float(np.linalg.norm(delta))


def obb_gap(a: OBB, b: OBB) -> float:
    a0, a1 = world_aabb(a)
    b0, b1 = world_aabb(b)
    return aabb_distance(a0, a1, b0, b1)


def contact_point(a: OBB, b: OBB) -> np.ndarray:
    a0, a1 = world_aabb(a)
    b0, b1 = world_aabb(b)
    mid = np.empty(3)
    for axis in range(3):
        lo = max(a0[axis], b0[axis])
        hi = min(a1[axis], b1[axis])
        if hi >= lo:
            mid[axis] = 0.5 * (lo + hi)
        elif a1[axis] < b0[axis]:
            mid[axis] = 0.5 * (a1[axis] + b0[axis])
        else:
            mid[axis] = 0.5 * (b1[axis] + a0[axis])
    return mid


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine],
            [y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine],
            [z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement],
        ]
    )


def pose(motion: Motion, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Map assembled points to the swept pose: ``p' = R @ p + translation``."""

    t = float(t)
    if motion.kind == "turn_then_lift":
        if t <= 0.5:
            angle = motion.angle * (t / 0.5)
            lift = 0.0
        else:
            angle = motion.angle
            lift = motion.distance * ((t - 0.5) / 0.5)
        rotation = rotation_matrix(motion.axis_dir, angle)
        origin = motion.axis_point
        translation = origin - rotation @ origin + motion.direction * lift
        return rotation, translation

    angle = motion.angle * t
    rotation = rotation_matrix(motion.axis_dir, angle) if abs(angle) > 0.0 else np.eye(3)
    if abs(motion.angle) > 0.0:
        origin = motion.axis_point
        translation = origin - rotation @ origin + motion.direction * motion.distance * t
    else:
        translation = motion.direction * motion.distance * t
    return rotation, translation


def transform_points(points: np.ndarray, motion: Motion, t: float) -> np.ndarray:
    rotation, translation = pose(motion, t)
    return points @ rotation.T + translation


def inverse_transform_points(points: np.ndarray, motion: Motion, t: float) -> np.ndarray:
    rotation, translation = pose(motion, t)
    return (points - translation) @ rotation


def transform_obb(box: OBB, motion: Motion, t: float) -> OBB:
    rotation, translation = pose(motion, t)
    return OBB(rotation @ box.center + translation, box.half.copy(), rotation @ box.axes)


def obb_overlap(a: OBB, b: OBB, eps: float = OVERLAP_EPS) -> bool:
    """True when the boxes penetrate by more than ``eps`` (face contact is allowed)."""

    axes = [a.axes[:, index] for index in range(3)]
    axes.extend(b.axes[:, index] for index in range(3))
    for i in range(3):
        for j in range(3):
            cross = np.cross(a.axes[:, i], b.axes[:, j])
            norm = float(np.linalg.norm(cross))
            if norm > 1e-8:
                axes.append(cross / norm)
    offset = b.center - a.center
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            continue
        direction = axis / norm
        radius_a = float(np.sum(a.half * np.abs(a.axes.T @ direction)))
        radius_b = float(np.sum(b.half * np.abs(b.axes.T @ direction)))
        distance = abs(float(np.dot(offset, direction)))
        if radius_a + radius_b - distance <= eps:
            return False
    return True


def parts_overlap(moving: list[OBB], static: list[OBB], eps: float = OVERLAP_EPS) -> bool:
    return any(obb_overlap(left, right, eps) for left in moving for right in static)


def sample_count(distance: float) -> int:
    return max(8, int(math.ceil(distance / 0.1)))


def sweep_clear(moving: list[OBB], static: list[OBB], motion: Motion, ground_z: float | None) -> bool:
    steps = sample_count(max(motion.distance, abs(motion.angle)))
    for step in range(1, steps + 1):
        t = step / steps
        posed = [transform_obb(box, motion, t) for box in moving]
        if ground_z is not None and any(obb_z_limit(box, -1.0) < ground_z - OVERLAP_EPS for box in posed):
            return False
        if parts_overlap(posed, static):
            return False
    return True


def clearance_distance(moving: Part, others: list[Part], direction: np.ndarray, gap: float = 0.5) -> float:
    direction = _unit(direction)
    own = obb_corners(moving.obbs[0])
    for box in moving.obbs[1:]:
        own = np.vstack([own, obb_corners(box)])
    min_self = float(np.min(own @ direction))
    if not others:
        span = float(np.max(own @ direction) - min_self)
        return span + gap
    other_pts = [obb_corners(box) for part in others for box in part.obbs]
    stacked = np.vstack(other_pts)
    max_other = float(np.max(stacked @ direction))
    return max(gap, max_other + gap - min_self)


def can_separate(moving: Part, static: Part, direction: np.ndarray, gap: float = 0.3) -> bool:
    """True when ``moving`` can translate off ``static`` without penetration."""

    direction = _unit(direction)
    distance = clearance_distance(moving, [static], direction, gap=gap)
    motion = Motion("translate", direction, distance, 0.0, part_com(moving), direction)
    if not sweep_clear(moving.obbs, static.obbs, motion, ground_z=None):
        return False
    start = min(obb_gap(a, b) for a in moving.obbs for b in static.obbs)
    end_boxes = [transform_obb(box, motion, 1.0) for box in moving.obbs]
    end = min(obb_gap(a, b) for a in end_boxes for b in static.obbs)
    return end > start + 0.05


def com_over_box(point: np.ndarray, box: OBB, eps: float = SUPPORT_EPS) -> bool:
    local = box.axes.T @ (point - box.center)
    return abs(float(local[0])) <= box.half[0] + eps and abs(float(local[1])) <= box.half[1] + eps


def part_supported(part: Part, others: list[Part], ground_z: float, eps: float = SUPPORT_EPS) -> bool:
    if part_min_z(part) <= ground_z + eps:
        return True
    com = part_com(part)
    for other in others:
        for box in other.obbs:
            if abs(part_min_z(part) - obb_z_limit(box, 1.0)) > eps:
                continue
            if com_over_box(com, box, eps):
                return True
    return False


def is_stable(parts: list[Part], ground_z: float) -> bool:
    for index, part in enumerate(parts):
        others = [other for other_index, other in enumerate(parts) if other_index != index]
        if not part_supported(part, others, ground_z):
            return False
    return True
