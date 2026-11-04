"""Build a 3MF package with OPC relationships to tasks and G-code."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from gnn_assembler.assembly import Assembly
from gnn_assembler.job.tasks_xml import AssemblyTask

_MODEL_TYPE = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"
_RELS_TYPE = "application/vnd.openxmlformats-package.relationships+xml"
_TASK_REL = "http://schemas.gnn-assembler.org/2026/relationships/tasks"
_GCODE_REL = "http://schemas.gnn-assembler.org/2026/relationships/gcode"
_MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"


def write_3mf(path: str | Path, assembly: Assembly, tasks: list[AssemblyTask], tasks_xml: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_3mf(assembly, tasks, tasks_xml))


def build_3mf(assembly: Assembly, tasks: list[AssemblyTask], tasks_xml: str) -> bytes:
    """Construct the package entirely in memory with deflated entries."""

    entries: dict[str, str] = {
        "[Content_Types].xml": _content_types(),
        "_rels/.rels": _package_rels(),
        "3D/3dmodel.model": _model_xml(assembly),
        "3D/_rels/3dmodel.model.rels": _model_rels(tasks),
        "Attachments/Tasks.xml": tasks_xml,
    }
    for task in tasks:
        entries[task.gcode_path] = task.gcode_body
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _content_types() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        f'  <Default Extension="model" ContentType="{_MODEL_TYPE}"/>\n'
        '  <Default Extension="xml" ContentType="text/xml"/>\n'
        '  <Default Extension="gcode" ContentType="text/plain"/>\n'
        f'  <Override PartName="/3D/3dmodel.model" ContentType="{_MODEL_TYPE}"/>\n'
        "</Types>\n"
    )


def _package_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        f'  <Relationship Target="/3D/3dmodel.model" Id="rel-model" Type="{_MODEL_REL}"/>\n'
        "</Relationships>\n"
    )


def _model_rels(tasks: list[AssemblyTask]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        f'  <Relationship Target="../Attachments/Tasks.xml" Id="rel-tasks" Type="{_TASK_REL}"/>',
    ]
    for task in tasks:
        filename = task.gcode_path.split("/")[-1]
        lines.append(
            f'  <Relationship Target="../Attachments/{filename}" Id="rel-gcode-{task.task_id}" Type="{_GCODE_REL}"/>'
        )
    lines.append("</Relationships>")
    return "\n".join(lines) + "\n"


def _model_xml(assembly: Assembly) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US"',
        '  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        '  <metadata name="Application">gnn-assembler</metadata>',
        "  <resources>",
    ]
    for object_id, part in enumerate(assembly.parts, start=1):
        lines.append(f'    <object id="{object_id}" type="model" name="{escape(part.name)}">')
        lines.append("      <mesh>")
        lines.append("        <vertices>")
        for vertex in part.vertices:
            lines.append(f'          <vertex x="{vertex[0]:.6f}" y="{vertex[1]:.6f}" z="{vertex[2]:.6f}"/>')
        lines.append("        </vertices>")
        lines.append("        <triangles>")
        for tri in part.faces:
            lines.append(f'          <triangle v1="{int(tri[0])}" v2="{int(tri[1])}" v3="{int(tri[2])}"/>')
        lines.append("        </triangles>")
        lines.append("      </mesh>")
        lines.append("    </object>")
    lines.append("  </resources>")
    lines.append("  <build>")
    for object_id in range(1, len(assembly.parts) + 1):
        lines.append(f'    <item objectid="{object_id}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>')
    lines.append("  </build>")
    lines.append("</model>")
    return "\n".join(lines) + "\n"
