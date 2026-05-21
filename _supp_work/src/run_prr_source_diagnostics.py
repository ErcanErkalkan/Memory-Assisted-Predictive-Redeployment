#!/usr/bin/env python3
"""Regenerate the PRR accepted-source diagnostic CSV files."""
from pathlib import Path
from dynamic_prr_experiment import Config, run_prr_source_diagnostics

if __name__ == "__main__":
    outdir = Path(__file__).resolve().parents[1] / "results"
    cfg = Config(runs=30, generations_per_state=3, workers=4, output_dir=str(outdir), mode="source-diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)
    run_prr_source_diagnostics(cfg, outdir)
    print(f"Wrote PRR source diagnostics to {outdir}")
