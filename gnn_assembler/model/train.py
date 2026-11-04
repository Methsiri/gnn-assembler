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
from gnn_assembler.planning.search import PlanError, plan_disassembly


@dataclass
class TrainResult:
    losses: list[float]
    scd: float
    connectivity: float
    path: Path
    log_path: Path
    skipped: int


def collect_examples(
    assemblies: list[Assembly],
    record: _RunLog,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[Assembly]]:
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    labeled: list[Assembly] = []
    total = len(assemblies)
    record(f"Labeling {total} assemblies with the physics planner")
    for index, assembly in enumerate(assemblies, start=1):
        try:
            graph = build_contact_graph(assembly)
            disassembly = plan_disassembly(assembly, graph, model=None)
            remaining = set(range(len(assembly.parts)))
            for step in disassembly:
                features, adjacency, indices = featurize_state(assembly, graph, remaining)
                target = np.zeros(len(indices), dtype=np.float32)
                target[indices.index(step.part_index)] = 1.0
                examples.append((features, adjacency, target))
                remaining.remove(step.part_index)
        except (PlanError, ValueError, RuntimeError) as exc:
            record(f"  skipped {index}/{total} {assembly.assembly_id}: {type(exc).__name__}: {exc}")
            continue
        labeled.append(assembly)
        record(f"  labeled {index}/{total} {assembly.assembly_id} ({len(disassembly)} steps)")
    if not examples:
        raise ValueError(f"No training examples could be labeled. See {record.path}")
    record(f"Labeled {len(labeled)}/{total} assemblies. Skipped {total - len(labeled)}.")
    return examples, labeled


def train_model(
    assemblies: list[Assembly],
    *,
    epochs: int = 20,
    out: str | Path = "checkpoints/heuristic.pt",
    device: str | None = None,
    seed: int = 0,
    lr: float = 1e-3,
    log: bool = False,
) -> TrainResult:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    chosen = select_device(device)
    path = Path(out)
    record = _RunLog(path.with_suffix(".log"), console=log)
    try:
        record(f"Training on {len(assemblies)} assemblies ({chosen})")
        examples, labeled = collect_examples(assemblies, record)
        model = RemovalGNN(FEATURE_DIM).to(chosen)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        history: list[float] = []
        model.train()
        record(f"Fitting {len(examples)} labeled steps for {epochs} epochs")
        for epoch in range(1, epochs + 1):
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
            mean_loss = total / len(examples)
            history.append(mean_loss)
            record(f"  epoch {epoch}/{epochs} loss {mean_loss:.4f}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "in_dim": FEATURE_DIM, "hidden": model.hidden},
            path,
        )
        record(f"Saved {path}. Scoring the labeled assemblies")
        scd, connectivity, score_skips = evaluate_loaded(labeled, model, chosen, record)
        skipped = len(assemblies) - len(labeled) + score_skips
        record(f"Done. Skipped {skipped}. Log {record.path}")
        return TrainResult(history, scd, connectivity, path, record.path, skipped)
    finally:
        record.close()


def evaluate_loaded(
    assemblies: list[Assembly],
    model: RemovalGNN,
    device: torch.device,
    record: _RunLog,
) -> tuple[float, float, int]:
    scds: list[float] = []
    cas: list[float] = []
    skipped = 0
    total = len(assemblies)
    for index, assembly in enumerate(assemblies, start=1):
        try:
            graph = build_contact_graph(assembly)
            disassembly = plan_disassembly(assembly, graph, model=model, device=str(device))
            scd, connectivity = evaluate_sequence(assembly, graph, disassembly)
        except (PlanError, ValueError, RuntimeError) as exc:
            skipped += 1
            record(f"  skipped score {index}/{total} {assembly.assembly_id}: {type(exc).__name__}: {exc}")
            continue
        scds.append(scd)
        cas.append(connectivity)
        record(f"  scored {index}/{total} {assembly.assembly_id} SCD {scd:.6f} CA {connectivity:.3f}")
    if not scds:
        raise ValueError(f"No assemblies could be scored. See {record.path}")
    return float(np.mean(scds)), float(np.mean(cas)), skipped


class _RunLog:
    def __init__(self, path: Path, *, console: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._console = console
        self._file = path.open("w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        self._file.write(message + "\n")
        self._file.flush()
        if self._console:
            print(message, flush=True)

    def close(self) -> None:
        self._file.close()


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
