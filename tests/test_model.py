"""Removal network shape and a short training run."""

from pathlib import Path

import torch

from gnn_assembler.data.ingest import discover
from gnn_assembler.data.samples import write_samples
from gnn_assembler.geometry.graph import FEATURE_DIM, build_contact_graph, featurize_state
from gnn_assembler.model.gnn import RemovalGNN
from gnn_assembler.model.train import load_model, train_model
from gnn_assembler.planning.search import plan_disassembly


def test_forward_scores_every_part(tmp_path: Path) -> None:
    write_samples(tmp_path / "samples")
    assembly = discover(tmp_path / "samples")[0]
    graph = build_contact_graph(assembly)
    features, adjacency, indices = featurize_state(assembly, graph, set(range(len(assembly.parts))))
    assert features.shape == (len(indices), FEATURE_DIM)
    model = RemovalGNN()
    logits = model(torch.tensor(features), torch.tensor(adjacency))
    assert logits.shape == (len(indices),)


def test_train_writes_a_checkpoint_and_still_plans(tmp_path: Path) -> None:
    write_samples(tmp_path / "samples")
    assemblies = discover(tmp_path / "samples")
    checkpoint = tmp_path / "heuristic.pt"
    result = train_model(assemblies, epochs=2, out=checkpoint, device="cpu", seed=0)
    assert checkpoint.exists()
    assert len(result.losses) == 2
    assert all(loss == loss for loss in result.losses)
    model, device = load_model(checkpoint, device="cpu")
    stack = next(item for item in assemblies if item.assembly_id == "stack")
    steps = plan_disassembly(stack, build_contact_graph(stack), model=model, device=str(device))
    assert [stack.parts[step.part_index].name for step in steps][0] == "top"
