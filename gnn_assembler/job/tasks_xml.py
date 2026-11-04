"""Serialize the forward assembly order as a task DAG."""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

import numpy as np

from gnn_assembler.assembly import Assembly
from gnn_assembler.geometry.collision import part_com
from gnn_assembler.geometry.graph import ContactGraph
from gnn_assembler.planning.search import RemovalStep, reverse_sequence


@dataclass
class AssemblyTask:
    task_id: int
    part_name: str
    part_index: int
    position: np.ndarray
    rotation_axis: np.ndarray
    rotation_angle: float
    insertion_direction: np.ndarray
    insertion_distance: float
    gcode_path: str
    gcode_body: str
    prerequisites: list[int]


def build_tasks(
    assembly: Assembly,
    graph: ContactGraph,
    disassembly: list[RemovalStep],
    programs: list[tuple[str, str]],
) -> list[AssemblyTask]:
    """``programs`` follows disassembly order and holds ``(zip path, gcode text)``."""

    if len(programs) != len(disassembly):
        raise ValueError("Each disassembly step needs one G-code program")
    by_part = {step.part_index: (step, programs[index]) for index, step in enumerate(disassembly)}
    tasks: list[AssemblyTask] = []
    placed: list[tuple[int, int]] = []
    for step in reverse_sequence(disassembly):
        motion = by_part[step.part_index][0].motion
        gcode_path, gcode_body = by_part[step.part_index][1]
        task_id = len(tasks) + 1
        prerequisites = [
            previous_id
            for previous_id, previous_index in placed
            if graph.pair(step.part_index, previous_index) is not None
        ]
        tasks.append(
            AssemblyTask(
                task_id=task_id,
                part_name=assembly.parts[step.part_index].name,
                part_index=step.part_index,
                position=part_com(assembly.parts[step.part_index]),
                rotation_axis=motion.axis_dir.copy(),
                rotation_angle=-motion.angle,
                insertion_direction=-motion.direction,
                insertion_distance=motion.distance,
                gcode_path=gcode_path,
                gcode_body=gcode_body,
                prerequisites=prerequisites,
            )
        )
        placed.append((task_id, step.part_index))
    return tasks


def tasks_to_xml(tasks: list[AssemblyTask]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<AssemblyJob>"]
    for task in tasks:
        position = task.position
        axis = task.rotation_axis
        direction = task.insertion_direction
        lines.append(f'  <Task id="{task.task_id}" part="{escape(task.part_name)}">')
        lines.append(
            f'    <Position x="{position[0]:.6f}" y="{position[1]:.6f}" z="{position[2]:.6f}"/>'
        )
        lines.append(
            "    <Rotation"
            f' x="{axis[0]:.6f}" y="{axis[1]:.6f}" z="{axis[2]:.6f}" angle="{task.rotation_angle:.6f}"/>'
        )
        lines.append(
            "    <Insertion"
            f' dx="{direction[0]:.6f}" dy="{direction[1]:.6f}" dz="{direction[2]:.6f}"'
            f' distance="{task.insertion_distance:.6f}"/>'
        )
        lines.append(f"    <GCode>{escape(task.gcode_path)}</GCode>")
        if task.prerequisites:
            lines.append("    <Prerequisites>")
            for prerequisite in task.prerequisites:
                lines.append(f'      <Prerequisite id="{prerequisite}"/>')
            lines.append("    </Prerequisites>")
        else:
            lines.append("    <Prerequisites/>")
        lines.append("  </Task>")
    lines.append("</AssemblyJob>")
    return "\n".join(lines) + "\n"
