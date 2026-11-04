"""Approach, align, and insert waypoints with a fixed residual policy."""

from __future__ import annotations

import numpy as np

from gnn_assembler.assembly import Part
from gnn_assembler.geometry.collision import Motion, part_com, transform_points

# Maps [direction_x, direction_y, direction_z, distance] to a small lateral bias.
_RESIDUAL = np.array(
    [
        [0.0, 0.012, 0.0, 0.0],
        [-0.012, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=float,
)


def residual_offset(direction: np.ndarray, distance: float) -> np.ndarray:
    """Fixed linear residual used in place of a contact-rich learned policy."""

    direction = np.asarray(direction, dtype=float).reshape(3)
    features = np.array([direction[0], direction[1], direction[2], float(distance)], dtype=float)
    return _RESIDUAL @ features


def insertion_waypoints(part: Part, extraction: Motion) -> list[np.ndarray]:
    """Sample the time-reversed extraction and bias only the approach point."""

    seated = part_com(part).reshape(1, 3)
    times = (1.0, 0.5, 0.0) if extraction.kind == "turn_then_lift" else (1.0, 0.15, 0.0)
    points = [transform_points(seated, extraction, t).reshape(3) for t in times]
    insertion_direction = -extraction.direction
    points[0] = points[0] + residual_offset(insertion_direction, extraction.distance)
    return points
