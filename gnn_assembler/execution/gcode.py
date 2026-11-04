"""Compile insertion waypoints to one G-code file per step."""

from __future__ import annotations

import uuid

import numpy as np


def render_gcode(part_name: str, task_id: int, waypoints: list[np.ndarray]) -> str:
    lines = [
        f"; gnn-assembler task {task_id} part {part_name}",
        "G21 ; millimeters",
        "G90 ; absolute",
    ]
    for index, point in enumerate(waypoints):
        x, y, z = (float(value) for value in np.asarray(point, dtype=float).reshape(3))
        if index == 0:
            lines.append(f"G0 X{x:.4f} Y{y:.4f} Z{z:.4f}")
        else:
            lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{z:.4f} F100")
    lines.append("M2")
    return "\n".join(lines) + "\n"


def new_gcode_name() -> str:
    return f"{uuid.uuid4()}.gcode"
