"""Subdivision and contact-graph tests."""

from pathlib import Path

from gnn_assembler.data.ingest import load_assembly
from gnn_assembler.data.samples import write_samples
from gnn_assembler.geometry.graph import build_contact_graph
from gnn_assembler.geometry.meshutil import load_mesh
from gnn_assembler.geometry.subdivide import subdivide_batch, subdivide_mesh


def test_subdivide_increases_vertex_count(tmp_path: Path) -> None:
    write_samples(tmp_path / "samples")
    source = tmp_path / "samples" / "multi_stack" / "base.obj"
    vertices, faces = load_mesh(source)
    dense_v, dense_f = subdivide_mesh(vertices, faces, levels=1)
    assert len(dense_v) > len(vertices)
    assert len(dense_f) == len(faces) * 4

    target = tmp_path / "dense"
    subdivide_batch(tmp_path / "samples", target, levels=1, num_proc=2)
    subdivided, _faces = load_mesh(target / "multi_stack" / "base.obj")
    assert len(subdivided) > len(vertices)
    assert (target / "multi_stack" / "assembly.json").exists()


def test_stack_contacts_and_joint_types(tmp_path: Path) -> None:
    write_samples(tmp_path / "samples")
    assembly = load_assembly(tmp_path / "samples" / "multi_stack")
    graph = build_contact_graph(assembly, use_sidecar=False)
    names = {assembly.parts[edge.a].name + "-" + assembly.parts[edge.b].name for edge in graph.edges}
    assert names == {"base-mid", "mid-top"}
    assert {edge.joint for edge in graph.edges} == {"prismatic"}

    peg = load_assembly(tmp_path / "samples" / "peg_hole")
    peg_graph = build_contact_graph(peg, use_sidecar=False)
    assert len(peg_graph.edges) == 1
    assert peg_graph.edges[0].joint == "prismatic"

    rotational = load_assembly(tmp_path / "samples" / "rotational")
    rotational_graph = build_contact_graph(rotational, use_sidecar=False)
    assert len(rotational_graph.edges) == 1
    assert rotational_graph.edges[0].joint == "revolute"
