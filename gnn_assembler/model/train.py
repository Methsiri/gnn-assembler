"""Supervise the removal network with physics-generated labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gnn_assembler.assembly import Assembly
from gnn_assembler.geometry.graph import FEATURE_DIM, build_contact_graph, featurize_state
from gnn_assembler.model.gnn import RemovalGNN, select_device
from gnn_assembler.model.metrics import evaluate_sequence
from gnn_assembler.planning.search import plan_disassembly


@dataclass
class TrainResult:
    losses: list[float]
    scd: float
    connectivity: float
    path: Path


def collect_examples(assemblies: list[Assembly]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for assembly in assemblies:
        graph = build_contact_graph(assembly)
        disassembly = plan_disassembly(assembly, graph, model=None)
        remaining = set(range(len(assembly.parts)))
        for step in disassembly:
            features, adjacency, indices = featurize_state(assembly, graph, remaining)
            target = np.zeros(len(indices), dtype=np.float32)
            target[indices.index(step.part_index)] = 1.0
            examples.append((features, adjacency, target))
            remaining.remove(step.part_index)
    if not examples:
        raise ValueError("No training examples could be labeled")
    return examples


def train_model(
    assemblies: list[Assembly],
    *,
    epochs: int = 20,
    out: str | Path = "checkpoints/heuristic.pt",
    device: str | None = None,
    seed: int = 0,
    lr: float = 1e-3,
) -> TrainResult:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chosen = select_device(device)
    examples = collect_examples(assemblies)
    model = RemovalGNN(FEATURE_DIM).to(chosen)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for features, adjacency, target in examples:
            optimizer.zero_grad()
            logits = model(
                torch.tensor(features, dtype=torch.float32, device=chosen),
                torch.tensor(adjacency, dtype=torch.float32, device=chosen),
            )
            loss = loss_fn(logits, torch.tensor(target, dtype=torch.float32, device=chosen))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        history.append(total / len(examples))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "in_dim": FEATURE_DIM, "hidden": model.hidden},
        path,
    )
    scd, connectivity = evaluate_loaded(assemblies, model, chosen)
    return TrainResult(history, scd, connectivity, path)


def evaluate_loaded(assemblies: list[Assembly], model: RemovalGNN, device: torch.device) -> tuple[float, float]:
    scds: list[float] = []
    cas: list[float] = []
    for assembly in assemblies:
        graph = build_contact_graph(assembly)
        disassembly = plan_disassembly(assembly, graph, model=model, device=str(device))
        scd, connectivity = evaluate_sequence(assembly, graph, disassembly)
        scds.append(scd)
        cas.append(connectivity)
    return float(np.mean(scds)), float(np.mean(cas))


def load_model(path: str | Path, device: str | None = None) -> tuple[RemovalGNN, torch.device]:
    chosen = select_device(device)
    try:
        blob = torch.load(path, map_location=chosen, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=chosen)
    model = RemovalGNN(in_dim=int(blob.get("in_dim", FEATURE_DIM)), hidden=int(blob.get("hidden", 32)))
    model.load_state_dict(blob["state_dict"])
    model.to(chosen)
    model.eval()
    return model, chosen
