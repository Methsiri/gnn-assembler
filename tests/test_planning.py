"""Physics search, reversal, and assembly metrics."""

from pathlib import Path

import numpy as np

from gnn_assembler.data.ingest import load_assembly
from gnn_assembler.data.samples import write_samples
from gnn_assembler.geometry.collision import inverse_transform_points, transform_points
from gnn_assembler.geometry.graph import build_contact_graph
from gnn_assembler.model.metrics import evaluate_sequence
from gnn_assembler.planning.search import plan_disassembly


def _samples(tmp_path: Path):
    write_samples(tmp_path / "samples")
    return tmp_path / "samples"


def test_stack_removes_top_down(tmp_path: Path) -> None:
    assembly = load_assembly(_samples(tmp_path) / "multi_stack")
    graph = build_contact_graph(assembly)
    steps = plan_disassembly(assembly, graph)
    order = [assembly.parts[step.part_index].name for step in steps]
    assert order == ["top", "mid", "base"]
    assert all(step.motion.kind == "translate" for step in steps)


def test_peg_lifts_out_before_the_base(tmp_path: Path) -> None:
    assembly = load_assembly(_samples(tmp_path) / "peg_hole")
    steps = plan_disassembly(assembly, build_contact_graph(assembly))
    assert assembly.parts[steps[0].part_index].name == "peg"
    assert steps[0].motion.angle == 0.0
    assert steps[0].motion.direction[2] > 0.5


def test_rotational_collar_uses_a_turn(tmp_path: Path) -> None:
    assembly = load_assembly(_samples(tmp_path) / "rotational")
    steps = plan_disassembly(assembly, build_contact_graph(assembly))
    assert assembly.parts[steps[0].part_index].name == "collar"
    assert abs(steps[0].motion.angle) > 0.5


def test_extraction_roundtrip_and_metrics(tmp_path: Path) -> None:
    assembly = load_assembly(_samples(tmp_path) / "multi_stack")
    graph = build_contact_graph(assembly)
    steps = plan_disassembly(assembly, graph)
    for step in steps:
        original = assembly.parts[step.part_index].vertices
        extracted = transform_points(original, step.motion, 1.0)
        restored = inverse_transform_points(extracted, step.motion, 1.0)
        assert np.allclose(restored, original, atol=1e-6)
    scd, connectivity = evaluate_sequence(assembly, graph, steps)
    assert scd < 1e-5
    assert connectivity == 1.0
