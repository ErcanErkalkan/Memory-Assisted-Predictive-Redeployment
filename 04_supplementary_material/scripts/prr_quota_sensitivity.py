#!/usr/bin/env python3
"""Targeted PRR source-quota sensitivity audit.

This script keeps the same benchmark, variation operators, and PRR admission filter
as the main experiment, but changes the relative PRR candidate-source shares. The
default protocol is an 8-scenario, 12-run targeted design-sensitivity audit. It is
not a replacement for the main 20-scenario, 30-run comparison.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse
import json
import pandas as pd

from dynamic_prr_experiment import Config, run_one

SCENARIO_SUBSET = [
    "FDA1", "FDA1-HF", "FDA1-HS", "dMOP2-HS",
    "DF1-HF", "DF4-HS", "APP-RESOURCE", "APP-SCHEDULING",
]

VARIANTS = {
    "default_35_30_25_10_split55": dict(prr_elite_ratio=0.35, prr_memory_ratio=0.30, prr_prediction_ratio=0.25, prr_diversity_ratio=0.10, prr_recent_memory_split=0.55),
    "elite_low_20": dict(prr_elite_ratio=0.20, prr_memory_ratio=0.30, prr_prediction_ratio=0.35, prr_diversity_ratio=0.15, prr_recent_memory_split=0.55),
    "elite_high_50": dict(prr_elite_ratio=0.50, prr_memory_ratio=0.22, prr_prediction_ratio=0.20, prr_diversity_ratio=0.08, prr_recent_memory_split=0.55),
    "memory_low_15": dict(prr_elite_ratio=0.40, prr_memory_ratio=0.15, prr_prediction_ratio=0.35, prr_diversity_ratio=0.10, prr_recent_memory_split=0.55),
    "memory_high_45": dict(prr_elite_ratio=0.25, prr_memory_ratio=0.45, prr_prediction_ratio=0.20, prr_diversity_ratio=0.10, prr_recent_memory_split=0.55),
    "prediction_low_10": dict(prr_elite_ratio=0.45, prr_memory_ratio=0.35, prr_prediction_ratio=0.10, prr_diversity_ratio=0.10, prr_recent_memory_split=0.55),
    "prediction_high_40": dict(prr_elite_ratio=0.25, prr_memory_ratio=0.25, prr_prediction_ratio=0.40, prr_diversity_ratio=0.10, prr_recent_memory_split=0.55),
    "diversity_low_02": dict(prr_elite_ratio=0.38, prr_memory_ratio=0.32, prr_prediction_ratio=0.28, prr_diversity_ratio=0.02, prr_recent_memory_split=0.55),
    "diversity_high_25": dict(prr_elite_ratio=0.25, prr_memory_ratio=0.25, prr_prediction_ratio=0.25, prr_diversity_ratio=0.25, prr_recent_memory_split=0.55),
    "recent_memory_heavy_80": dict(prr_elite_ratio=0.35, prr_memory_ratio=0.30, prr_prediction_ratio=0.25, prr_diversity_ratio=0.10, prr_recent_memory_split=0.80),
    "transfer_memory_heavy_25": dict(prr_elite_ratio=0.35, prr_memory_ratio=0.30, prr_prediction_ratio=0.25, prr_diversity_ratio=0.10, prr_recent_memory_split=0.25),
}

METRICS = ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]


def average_ranks(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    means = raw.groupby(["scenario", "variant"]).mean(numeric_only=True).reset_index()
    for metric in METRICS:
        wide = means.pivot(index="scenario", columns="variant", values=metric)
        ascending = metric in {"IGD", "DecisionDrift"}
        ranks = wide.rank(axis=1, method="average", ascending=ascending)
        for variant, val in ranks.mean().items():
            rows.append({"metric": metric, "variant": variant, "average_rank": float(val)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../results/quota_sensitivity")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--states", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Config(runs=args.runs, pop_size=args.pop_size, variables=10, states=args.states, generations_per_state=args.generations, workers=1)
    rows = []
    for variant, params in VARIANTS.items():
        cfg = replace(base, **params)
        for scenario in SCENARIO_SUBSET:
            for run in range(args.runs):
                rec = run_one(scenario, "PRR-NSGA-II", run, cfg)
                rec.update({"variant": variant, **params})
                rows.append(rec)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "prr_quota_sensitivity_raw.csv", index=False)
    summary = raw.groupby("variant").agg(
        HV_mean=("HV", "mean"), HV_std=("HV", "std"),
        IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"),
        Stability_mean=("Stability", "mean"), Stability_std=("Stability", "std"),
        DecisionDrift_mean=("DecisionDrift", "mean"), DecisionDrift_std=("DecisionDrift", "std"),
        QDR_mean=("QDR", "mean"), QDR_std=("QDR", "std"),
        elite_ratio=("prr_elite_ratio", "first"), memory_ratio=("prr_memory_ratio", "first"),
        prediction_ratio=("prr_prediction_ratio", "first"), diversity_ratio=("prr_diversity_ratio", "first"),
        recent_memory_split=("prr_recent_memory_split", "first"),
    ).reset_index().sort_values("HV_mean", ascending=False)
    summary.to_csv(out / "prr_quota_sensitivity_summary.csv", index=False)
    ranks = average_ranks(raw)
    ranks.to_csv(out / "prr_quota_sensitivity_average_ranks.csv", index=False)
    # Compact manuscript table: show selected variants and rank indicators.
    piv = ranks.pivot(index="variant", columns="metric", values="average_rank").reset_index()
    compact = summary.merge(piv, on="variant", suffixes=("", "_rank"))
    compact.to_csv(out / "prr_quota_sensitivity_compact.csv", index=False)

    metadata = {
        "audit_type": "targeted_source_quota_sensitivity",
        "purpose": "design-sensitivity diagnostic, not the main all-scenario comparison",
        "scenarios": SCENARIO_SUBSET,
        "scenario_count": len(SCENARIO_SUBSET),
        "variant_count": len(VARIANTS),
        "runs_per_scenario_variant": args.runs,
        "states": args.states,
        "generations_per_state": args.generations,
        "pop_size": args.pop_size,
        "algorithm": "PRR-NSGA-II",
        "raw_rows": int(len(raw)),
        "expected_raw_rows": int(len(SCENARIO_SUBSET) * len(VARIANTS) * args.runs),
    }
    (out / "prr_quota_sensitivity_protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (out / "README_PRR_QUOTA_SENSITIVITY_PROTOCOL.txt").write_text(
        "PRR quota-sensitivity audit protocol\n"
        "====================================\n"
        "This directory contains a targeted design-sensitivity audit, not the main 20-scenario, 30-run comparison.\n"
        f"Protocol: {len(SCENARIO_SUBSET)} representative scenarios, {len(VARIANTS)} quota variants, {args.runs} independent runs per scenario/variant, "
        f"{args.states} environmental states, and {args.generations} generations per state.\n"
        "The manuscript-facing table reports means over this 8-scenario, 12-run targeted audit.\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
