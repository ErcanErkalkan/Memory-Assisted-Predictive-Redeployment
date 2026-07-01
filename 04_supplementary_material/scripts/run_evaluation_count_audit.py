#!/usr/bin/env python3
"""Dedicated wrapper for the objective-evaluation accounting audit."""
from __future__ import annotations
import argparse
from pathlib import Path

from dynamic_prr_experiment import Config, run_evaluation_count_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate evaluation-count audit CSV files.")
    parser.add_argument("--out-dir", default="../results", help="Directory for audit CSV outputs.")
    parser.add_argument("--runs", type=int, default=30, help="Main-experiment run count stored in metadata/config context.")
    parser.add_argument("--audit-runs", type=int, default=5, help="Counted runs per scenario-algorithm pair.")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--variables", type=int, default=10)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--generations", type=int, default=3, help="Generations per environmental state.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        runs=args.runs,
        pop_size=args.pop_size,
        variables=args.variables,
        states=args.states,
        generations_per_state=args.generations,
        workers=args.workers,
        output_dir=str(outdir),
        mode="evaluation-count",
    )
    run_evaluation_count_audit(cfg, outdir, audit_runs=args.audit_runs)


if __name__ == "__main__":
    main()
