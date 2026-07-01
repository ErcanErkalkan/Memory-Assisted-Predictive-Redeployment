#!/usr/bin/env python3
"""Synthetic admission-control stress test for PRR versus direct insertion.

The test is intentionally simple and diagnostic. It creates objective-space
candidate pools with non-predicted candidates near a moving front and predicted
candidates corrupted by a controllable directional error. PRR accepts predicted
candidates only through environmental selection; direct insertion reserves a fixed
prediction quota. The reported contamination rate is the fraction of retained
predicted candidates whose priority is worse than the Nth selected non-predicted
candidate from the same pool.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from dynamic_prr_experiment import front_ranks_2d, crowding_distance


def priority_order(F: np.ndarray) -> np.ndarray:
    ranks = front_ranks_2d(F)
    cd = crowding_distance(F, ranks)
    return np.lexsort((-cd, ranks))


def select_indices(F: np.ndarray, N: int) -> np.ndarray:
    return priority_order(F)[:N]


def simulate(noise: float, seed: int, N: int = 100, q_pred: int = 25, pool_multiplier: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    n_total = N * pool_multiplier
    n_pred = int(round(0.25 * n_total))
    n_other = n_total - n_pred
    # Other candidates: imperfect but centered around a convex Pareto-like curve.
    u = rng.random(n_other)
    F_other = np.column_stack([u, 1.0 - np.sqrt(u)]) + rng.normal(0, 0.025, size=(n_other, 2))
    F_other = np.clip(F_other, 0, None)
    # Predicted candidates: accurate at noise=0, increasingly shifted upward/right.
    up = rng.random(n_pred)
    F_pred = np.column_stack([up, 1.0 - np.sqrt(up)]) + rng.normal(0, 0.025, size=(n_pred, 2))
    harmful_shift = np.abs(rng.normal(noise, noise * 0.35 + 1e-9, size=(n_pred, 2)))
    F_pred = np.clip(F_pred + harmful_shift, 0, None)
    F = np.vstack([F_other, F_pred])
    labels = np.array(["other"] * n_other + ["prediction"] * n_pred)

    # PRR: selection-filtered admission over the whole current-state pool.
    prr_idx = select_indices(F, N)
    prr_pred = int(np.sum(labels[prr_idx] == "prediction"))

    # Direct insertion: keep q predicted candidates first, then fill from other candidates.
    pred_idx = np.where(labels == "prediction")[0]
    pred_order = pred_idx[priority_order(F[pred_idx])[:q_pred]]
    other_idx = np.where(labels == "other")[0]
    other_fill = other_idx[priority_order(F[other_idx])[:N - len(pred_order)]]
    dir_idx = np.concatenate([pred_order, other_fill])
    dir_pred = int(np.sum(labels[dir_idx] == "prediction"))

    # A retained predicted candidate is contaminated if it is worse than the Nth best
    # non-predicted candidate available in the same pool.
    other_priority = other_idx[priority_order(F[other_idx])]
    threshold_set = set(other_priority[:min(N, len(other_priority))].tolist())
    threshold_rank = {idx: r for r, idx in enumerate(priority_order(F))}
    worst_other_rank = max(threshold_rank[i] for i in threshold_set)
    prr_contam = sum(1 for i in prr_idx if labels[i] == "prediction" and threshold_rank[i] > worst_other_rank)
    dir_contam = sum(1 for i in dir_idx if labels[i] == "prediction" and threshold_rank[i] > worst_other_rank)

    # Normalized priority loss: smaller is better.
    prr_loss = float(np.mean([threshold_rank[i] for i in prr_idx]) / len(F))
    dir_loss = float(np.mean([threshold_rank[i] for i in dir_idx]) / len(F))
    return {
        "noise": noise, "seed": seed,
        "PRR_predicted_accepted": prr_pred,
        "DIR_predicted_forced": dir_pred,
        "PRR_contaminated_predictions": prr_contam,
        "DIR_contaminated_predictions": dir_contam,
        "PRR_priority_loss": prr_loss,
        "DIR_priority_loss": dir_loss,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../results/prediction_noise_stress")
    ap.add_argument("--replicates", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for noise in [0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30]:
        for r in range(args.replicates):
            rows.append(simulate(noise, seed=10000 + r))
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "prediction_noise_stress_raw.csv", index=False)
    summary = raw.groupby("noise").agg(
        PRR_predicted_accepted_mean=("PRR_predicted_accepted", "mean"),
        DIR_predicted_forced_mean=("DIR_predicted_forced", "mean"),
        PRR_contaminated_predictions_mean=("PRR_contaminated_predictions", "mean"),
        DIR_contaminated_predictions_mean=("DIR_contaminated_predictions", "mean"),
        PRR_priority_loss_mean=("PRR_priority_loss", "mean"),
        DIR_priority_loss_mean=("DIR_priority_loss", "mean"),
    ).reset_index()
    summary.to_csv(out / "prediction_noise_stress_summary.csv", index=False)

if __name__ == "__main__":
    main()
