"""Task DAG, G-code, and 3MF packaging."""

import zipfile
from pathlib import Path
from xml.etree import ElementTree

from gnn_assembler.cli import main
from gnn_assembler.data.ingest import load_assembly
from gnn_assembler.data.samples import write_samples
from gnn_assembler.execution.trajectory import residual_offset
from gnn_assembler.pipeline import build_job


def test_residual_is_a_small_fixed_bias() -> None:
    direction = [0.0, 0.0, -1.0]
    first = residual_offset(direction, 1.5)
    second = residual_offset(direction, 1.5)
    assert first.tolist() == second.tolist()
    assert float(abs(first).max()) < 0.05


def test_stack_tasks_form_a_contact_dag(tmp_path: Path) -> None:
    write_samples(tmp_path / "samples")
    assembly = load_assembly(tmp_path / "samples" / "multi_stack")
    output = tmp_path / "stack.3mf"
    job = build_job(assembly, levels=1, output=output)
    assert job.disassembly_order == ["top", "mid", "base"]
    assert job.assembly_order == ["base", "mid", "top"]
    assert job.scd < 1e-5
    assert job.connectivity == 1.0
    by_name = {task.part_name: task for task in job.tasks}
    assert by_name["base"].prerequisites == []
    assert by_name["mid"].prerequisites == [by_name["base"].task_id]
    assert by_name["top"].prerequisites == [by_name["mid"].task_id]
    gcode_names = [task.gcode_path for task in job.tasks]
    assert len(gcode_names) == len(set(gcode_names))
    assert all(f"task {task.task_id} part {task.part_name}" in task.gcode_body for task in job.tasks)
    assert all("G21" in task.gcode_body and "G90" in task.gcode_body for task in job.tasks)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "[Content_Types].xml" in names
        assert "_rels/.rels" in names
        assert "3D/3dmodel.model" in names
        assert "3D/_rels/3dmodel.model.rels" in names
        assert "Attachments/Tasks.xml" in names
        for task in job.tasks:
            assert task.gcode_path in names
        info = archive.getinfo("3D/3dmodel.model")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        rels = archive.read("3D/_rels/3dmodel.model.rels").decode()
        assert "Tasks.xml" in rels
        for task in job.tasks:
            assert Path(task.gcode_path).name in rels
        model = archive.read("3D/3dmodel.model").decode()
        assert model.count("<object ") == 3
        root = ElementTree.fromstring(archive.read("Attachments/Tasks.xml"))
    task_nodes = list(root)
    assert [node.attrib["part"] for node in task_nodes] == ["base", "mid", "top"]
    assert task_nodes[0].find("Prerequisites").find("Prerequisite") is None
    assert task_nodes[1].find("Prerequisites").find("Prerequisite").attrib["id"] == "1"


def test_plan_cli_prints_metrics(tmp_path: Path, capsys) -> None:
    samples = tmp_path / "samples"
    write_samples(samples)
    output = tmp_path / "job.3mf"
    code = main(
        [
            "plan",
            "--dir",
            str(samples),
            "--id",
            "stack",
            "--output",
            str(output),
            "--levels",
            "0",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Disassembly: top -> mid -> base" in printed
    assert "SCD:" in printed
    assert output.exists()
