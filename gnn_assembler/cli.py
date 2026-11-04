"""Command line for subdivision, training, and 3MF job export."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnn_assembler.data.ingest import discover, find_assembly
from gnn_assembler.data.samples import write_samples
from gnn_assembler.geometry.subdivide import subdivide_batch
from gnn_assembler.model.train import load_model, train_model
from gnn_assembler.pipeline import build_job, describe_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gnn-assemble", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-samples", help="Write the built-in peg, rotational, and stack assemblies")
    generate.add_argument("--out", default="samples", type=Path)

    subdivide = sub.add_parser("subdivide", help="Midpoint-subdivide meshes in an assembly or dataset directory")
    subdivide.add_argument("--source", required=True, type=Path)
    subdivide.add_argument("--target", required=True, type=Path)
    subdivide.add_argument("--levels", type=int, default=1)
    subdivide.add_argument("--num-proc", type=int, default=1)

    train = sub.add_parser("train", help="Train the removal heuristic on physics-labeled assemblies")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--out", default=Path("checkpoints/heuristic.pt"), type=Path)
    train.add_argument("--device", default=None)

    plan = sub.add_parser("plan", help="Plan an assembly and write a 3MF job")
    plan.add_argument("--dir", required=True, type=Path)
    plan.add_argument("--id", required=True)
    plan.add_argument("--output", required=True, type=Path)
    plan.add_argument("--checkpoint", default=None, type=Path)
    plan.add_argument("--levels", type=int, default=1)
    plan.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "generate-samples":
            write_samples(args.out)
            print(f"Wrote sample assemblies to {args.out}")
            return 0
        if args.command == "subdivide":
            subdivide_batch(args.source, args.target, levels=args.levels, num_proc=args.num_proc)
            print(f"Subdivided {args.source} into {args.target}")
            return 0
        if args.command == "train":
            assemblies = discover(args.data)
            result = train_model(
                assemblies,
                epochs=args.epochs,
                out=args.out,
                device=args.device,
                log=True,
            )
            print(
                f"Saved {result.path} after {args.epochs} epochs. "
                f"Final loss {result.losses[-1]:.4f}. SCD {result.scd:.6f}. CA {result.connectivity:.3f}. "
                f"Skipped {result.skipped}. Log {result.log_path}."
            )
            return 0
        if args.command == "plan":
            assembly = find_assembly(args.dir, args.id)
            model = None
            device = "cpu"
            if args.checkpoint is not None:
                model, chosen = load_model(args.checkpoint, args.device)
                device = str(chosen)
            elif args.device:
                device = args.device
            job = build_job(
                assembly,
                model=model,
                device=device,
                levels=args.levels,
                output=args.output,
            )
            print(describe_job(job))
            return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    raise AssertionError(args.command)
