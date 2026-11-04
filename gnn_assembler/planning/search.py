"""GNN-guided breadth-first search over removal order, then time reversal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from gnn_assembler.assembly import Assembly
from gnn_assembler.geometry.collision import Motion, part_com
from gnn_assembler.geometry.graph import ContactGraph, featurize_state
from gnn_assembler.planning.physics import find_removal_motion


class PlanError(RuntimeError):
    """Raised when no physically feasible disassembly exists."""


@dataclass
class RemovalStep:
    part_index: int
    motion: Motion


def rank_parts(
    assembly: Assembly,
    graph: ContactGraph,
    remaining: set[int],
    model=None,
    device: str = "cpu",
) -> list[int]:
    """Order candidate parts. Without a model, higher parts are tried first."""

    if model is None:
        return sorted(
            remaining,
            key=lambda index: (-float(part_com(assembly.parts[index])[2]), assembly.parts[index].name),
        )
    import torch

    features, adjacency, indices = featurize_state(assembly, graph, remaining)
    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(feature_tensor, adjacency_tensor).detach().cpu().numpy()
    order = np.argsort(-logits)
    return [indices[int(row)] for row in order]


def plan_disassembly(
    assembly: Assembly,
    graph: ContactGraph,
    model=None,
    device: str = "cpu",
    max_states: int = 5000,
) -> list[RemovalStep]:
    """Breadth-first search. The learned scores only change which child is tried first."""

    if not assembly.parts:
        raise PlanError("Assembly has no parts")
    start = frozenset(range(len(assembly.parts)))
    parent: dict[frozenset[int], tuple[frozenset[int], int, Motion] | None] = {start: None}
    queue: deque[frozenset[int]] = deque([start])
    goal: frozenset[int] | None = None
    expanded = 0
    while queue:
        state = queue.popleft()
        if not state:
            goal = state
            break
        expanded += 1
        if expanded > max_states:
            break
        for part_index in rank_parts(assembly, graph, set(state), model, device):
            motion = find_removal_motion(assembly, graph, set(state), part_index)
            if motion is None:
                continue
            nxt = state - {part_index}
            if nxt in parent:
                continue
            parent[nxt] = (state, part_index, motion)
            if not nxt:
                goal = nxt
                queue.clear()
                break
            queue.append(nxt)
    if goal is None:
        greedy = _greedy(assembly, graph, model, device)
        if greedy is None:
            raise PlanError(f"No feasible disassembly for {assembly.assembly_id}")
        return greedy
    return _reconstruct(parent, goal)


def reverse_sequence(disassembly: list[RemovalStep]) -> list[RemovalStep]:
    """Time-reverse a disassembly into the forward assembly order.

    Each stored motion remains the extraction from the seated pose. Callers that
    need the insertion direction negate ``direction`` and ``angle``. Sampling
    ``t`` from 1 down to 0 replays the forward path.
    """

    return [RemovalStep(step.part_index, step.motion) for step in reversed(disassembly)]


def _reconstruct(
    parent: dict[frozenset[int], tuple[frozenset[int], int, Motion] | None],
    goal: frozenset[int],
) -> list[RemovalStep]:
    steps: list[RemovalStep] = []
    state: frozenset[int] | None = goal
    while state is not None and parent[state] is not None:
        previous, part_index, motion = parent[state]
        steps.append(RemovalStep(part_index, motion))
        state = previous
    steps.reverse()
    return steps


def _greedy(assembly, graph, model, device) -> list[RemovalStep] | None:
    remaining = set(range(len(assembly.parts)))
    steps: list[RemovalStep] = []
    while remaining:
        moved = False
        for part_index in rank_parts(assembly, graph, remaining, model, device):
            motion = find_removal_motion(assembly, graph, remaining, part_index)
            if motion is None:
                continue
            steps.append(RemovalStep(part_index, motion))
            remaining.remove(part_index)
            moved = True
            break
        if not moved:
            return None
    return steps
