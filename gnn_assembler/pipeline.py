"""End-to-end assembly job: subdivide, plan, emit tasks, pack a 3MF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gnn_assembler.assembly import Assembly, Part
from gnn_assembler.execution.gcode import new_gcode_name, render_gcode
from gnn_assembler.execution.trajectory import insertion_waypoints
from gnn_assembler.geometry.graph import build_contact_graph
from gnn_assembler.geometry.subdivide import subdivide_mesh
from gnn_assembler.job.tasks_xml import AssemblyTask, build_tasks, tasks_to_xml
from gnn_assembler.job.threemf import write_3mf
from gnn_assembler.model.metrics import evaluate_sequence
from gnn_assembler.planning.search import plan_disassembly, reverse_sequence


@dataclass
class AssemblyJob:
    assembly_id: str
    disassembly_order: list[str]
    assembly_order: list[str]
    scd: float
    connectivity: float
    tasks: list[AssemblyTask]
    tasks_xml: str
    output_path: Path | None


def prepare_assembly(assembly: Assembly, levels: int) -> Assembly:
    if levels <= 0:
        return assembly
    parts: list[Part] = []
    for part in assembly.parts:
        vertices, faces = subdivide_mesh(part.vertices, part.faces, levels)
        parts.append(Part(part.name, vertices, faces, part.obbs, part.mass))
    return assembly.renamed(parts)


def build_job(
    assembly: Assembly,
    *,
    model=None,
    device: str = "cpu",
    levels: int = 1,
    output: str | Path | None = None,
) -> AssemblyJob:
    prepared = prepare_assembly(assembly, levels)
    contact = build_contact_graph(prepared)
    disassembly = plan_disassembly(prepared, contact, model=model, device=device)
    waypoints = {
        step.part_index: insertion_waypoints(prepared.parts[step.part_index], step.motion)
        for step in disassembly
    }
    programs: list[tuple[str, str] | None] = [None] * len(disassembly)
    disassembly_index = {step.part_index: index for index, step in enumerate(disassembly)}
    for task_id, step in enumerate(reverse_sequence(disassembly), start=1):
        part = prepared.parts[step.part_index]
        body = render_gcode(part.name, task_id, waypoints[step.part_index])
        programs[disassembly_index[step.part_index]] = (f"Attachments/{new_gcode_name()}", body)
    typed_programs = [item for item in programs if item is not None]
    tasks = build_tasks(prepared, contact, disassembly, typed_programs)
    xml = tasks_to_xml(tasks)
    scd, connectivity = evaluate_sequence(prepared, contact, disassembly)
    output_path = Path(output) if output is not None else None
    if output_path is not None:
        write_3mf(output_path, prepared, tasks, xml)
    return AssemblyJob(
        assembly_id=prepared.assembly_id,
        disassembly_order=[prepared.parts[step.part_index].name for step in disassembly],
        assembly_order=[task.part_name for task in tasks],
        scd=scd,
        connectivity=connectivity,
        tasks=tasks,
        tasks_xml=xml,
        output_path=output_path,
    )


def describe_job(job: AssemblyJob) -> str:
    lines = [
        f"Assembly: {job.assembly_id}",
        f"Disassembly: {' -> '.join(job.disassembly_order)}",
        f"Assembly: {' -> '.join(job.assembly_order)}",
        f"SCD: {job.scd:.6f}",
        f"CA: {job.connectivity:.3f}",
    ]
    if job.output_path is not None:
        lines.append(f"3MF: {job.output_path}")
    return "\n".join(lines)
