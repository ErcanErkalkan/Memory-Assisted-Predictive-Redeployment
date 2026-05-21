#!/usr/bin/env python3
"""Reproducible dynamic multi-objective optimization experiments for PRR-NSGA-II.

The implementation uses explicit two-objective dynamic benchmark-family scenarios and
multiple evolutionary dynamic response strategies. It is intentionally self-contained and depends
only on NumPy, pandas, matplotlib, and scipy for statistical tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, friedmanchisquare
except Exception:  # pragma: no cover
    mannwhitneyu = None
    friedmanchisquare = None


@dataclass
class EvaluationCounter:
    """Counts explicit objective-vector evaluations performed by eval_objectives().

    A function call is one invocation of eval_objectives(); a vector evaluation is
    one candidate row evaluated inside that invocation. The counter is activated
    through evaluation_accounting() so normal experiments are unaffected.
    """
    objective_vector_evaluations: int = 0
    objective_function_calls: int = 0

    def record(self, X: np.ndarray) -> None:
        self.objective_function_calls += 1
        self.objective_vector_evaluations += int(X.shape[0])


_ACTIVE_EVALUATION_COUNTER: EvaluationCounter | None = None


@contextmanager
def evaluation_accounting(counter: EvaluationCounter | None = None):
    """Enable objective-evaluation accounting within the current process.

    The hook is intentionally scoped. Reference-front construction calls
    eval_objectives(..., count=False) so the audit reports algorithm-side
    objective-vector evaluations rather than metric/reference-grid evaluations.
    """
    global _ACTIVE_EVALUATION_COUNTER
    previous = _ACTIVE_EVALUATION_COUNTER
    active = counter if counter is not None else EvaluationCounter()
    _ACTIVE_EVALUATION_COUNTER = active
    try:
        yield active
    finally:
        _ACTIVE_EVALUATION_COUNTER = previous

ALGORITHMS = [
    "NSGA-II",
    "DNSGA-II",
    "MNSGA-II",
    "PPS-NSGA-II",
    "DMOEA/D",
    "ARVLP-NSGA-II",
    "KTR-NSGA-II",
    "ABR-NSGA-II",
    "RDMOEA-NSGA-II",
    "PRR-NSGA-II",
]
ABLATION_ALGORITHMS = [
    "PRR-complete",
    "PRR-no-prediction",
    "PRR-no-memory",
    "PRR-no-diversity",
    "PRR-direct-insertion",
    "PRR-no-elite",
]
SCENARIOS = [
    "FDA1",
    "FDA2",
    "FDA3",
    "dMOP1",
    "dMOP2",
    "dMOP3",
    "DF1",
    "DF2",
    "DF3",
    "DF4",
    "DF5",
    "FDA1-HF",
    "FDA1-HS",
    "dMOP2-LF",
    "dMOP2-HS",
    "DF1-HF",
    "DF4-HS",
    "DF5-HF",
    "APP-RESOURCE",
    "APP-SCHEDULING",
]

@dataclass
class Config:
    runs: int = 30
    pop_size: int = 100
    variables: int = 10
    states: int = 20
    # Manuscript protocol default: 3 generations per environmental state.
    # Larger budgets are explored explicitly in the generation-sensitivity audit.
    generations_per_state: int = 3
    offspring_multiplier: int = 1
    memory_size: int = 100
    # Number of recent non-empty centroids retained for prediction.
    # With prediction_window=4, the movement estimate uses at most 3
    # consecutive centroid displacement vectors, matching the manuscript notation.
    prediction_window: int = 4
    prediction_scale: float = 0.85
    immigrant_ratio: float = 0.18
    mutation_sigma: float = 0.07
    crossover_rate: float = 0.9
    # PRR ratios are normalized generation shares for the over-generated
    # response pool, not admission quotas. With prr_pool_multiplier=2.0 and
    # pop_size=N, the default proposal counts are approximately 0.70N elite,
    # 0.60N memory/transfer, 0.50N prediction, and 0.20N diversity; NSGA-II
    # environmental selection then filters this 2N pool down to N solutions.
    prr_elite_ratio: float = 0.35
    prr_memory_ratio: float = 0.30
    prr_prediction_ratio: float = 0.25
    prr_diversity_ratio: float = 0.10
    prr_recent_memory_split: float = 0.55
    prr_pool_multiplier: float = 2.0
    output_dir: str = "."
    mode: str = "main"
    workers: int = 0

def stable_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def env_time(state: int, states: int) -> float:
    if states <= 1:
        return 0.0
    return state / (states - 1)

def scenario_base(scenario: str) -> str:
    """Return the base problem name without severity/frequency suffixes."""
    aliases = {
        "DYN-FDA1": "FDA1",
        "DYN-FDA2": "FDA2",
        "DYN-FDA3": "FDA3",
        "DYN-dMOP1": "dMOP1",
        "DYN-dMOP2": "dMOP2",
        "DYN-dMOP3": "dMOP3",
        "DYN-DF1": "DF1",
        "DYN-DF2": "DF2",
        "DYN-DF3": "DF3",
    }
    if scenario in aliases:
        return aliases[scenario]
    for suffix in ("-HF", "-LF", "-HS", "-LS"):
        if scenario.endswith(suffix):
            return scenario[: -len(suffix)]
    return scenario

def scenario_profile(scenario: str) -> Tuple[float, float]:
    """Return change-frequency and change-severity multipliers."""
    frequency = 1.0
    severity = 1.0
    if scenario.endswith("-HF"):
        frequency = 2.0
    elif scenario.endswith("-LF"):
        frequency = 0.5
    if scenario.endswith("-HS"):
        severity = 1.35
    elif scenario.endswith("-LS"):
        severity = 0.65
    return frequency, severity

def benchmark_time(scenario: str, state: int, states: int) -> float:
    frequency, _ = scenario_profile(scenario)
    return frequency * env_time(state, states)

def dynamic_signal(scenario: str, state: int, states: int) -> float:
    _, severity = scenario_profile(scenario)
    signal = math.sin(0.5 * math.pi * benchmark_time(scenario, state, states))
    return float(np.clip(severity * signal, -1.0, 1.0))

def decode_unit(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + (upper - lower) * x

def app_shift(scenario: str, tau: float) -> float:
    if scenario == "APP-RESOURCE":
        return 0.35 + 0.22 * math.sin(2 * math.pi * tau + 0.4) + 0.05 * math.sin(9 * math.pi * tau)
    if scenario == "APP-SCHEDULING":
        return 0.40 + 0.18 * math.cos(2 * math.pi * tau + 0.6) + 0.04 * math.sin(8 * math.pi * tau)
    return 0.35

def eval_objectives(x: np.ndarray, scenario: str, state: int, states: int, *, count: bool = True) -> np.ndarray:
    """Two-objective minimization functions for the implemented dynamic benchmark suites.

    When evaluation_accounting() is active, this function records one objective
    function call and the number of candidate vectors evaluated. Calls used only
    to construct reference fronts pass count=False and are excluded from the
    budget-fairness audit.
    """
    tau = env_time(state, states)
    base = scenario_base(scenario)
    v = dynamic_signal(scenario, state, states)
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if count and _ACTIVE_EVALUATION_COUNTER is not None:
        _ACTIVE_EVALUATION_COUNTER.record(X)
    x0 = np.clip(X[:, 0], 0.0, 1.0)
    tail = X[:, 1:]
    eps = 1e-12

    if base == "APP-RESOURCE":
        sh = app_shift(base, tau)
        price = 0.50 + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * tau + 0.7))
        target = np.linspace(sh - 0.08, sh + 0.08, X.shape[1] - 1)
        balance = np.mean((tail - target) ** 2, axis=1)
        cost = 0.35 * x0 + 0.55 * price * np.mean(tail, axis=1) + 0.10 * np.var(tail, axis=1)
        emissions = (1.0 - x0) ** 2 + 1.8 * balance + 0.10 * np.sin(2 * math.pi * tau) ** 2
        return np.column_stack([np.clip(cost, 0, 1.8), np.clip(emissions, 0, 2.4)])
    if base == "APP-SCHEDULING":
        sh = app_shift(base, tau)
        urgency = 0.45 + 0.30 * (0.5 + 0.5 * math.cos(2 * math.pi * tau))
        target = sh + 0.10 * np.sin(np.arange(1, X.shape[1]) * math.pi * tau)
        lateness = np.mean(np.maximum(tail - target, 0.0) ** 2, axis=1)
        disruption = np.mean(np.abs(tail - target), axis=1)
        makespan = 0.45 * x0 + urgency * lateness + 0.12 * np.mean(tail, axis=1)
        switching = (1.0 - x0) ** 1.5 + 1.35 * disruption
        return np.column_stack([np.clip(makespan, 0, 1.8), np.clip(switching, 0, 2.4)])

    if base == "FDA1":
        y = decode_unit(tail, -1.0, 1.0)
        g = 1.0 + np.sum((y - v) ** 2, axis=1)
        f1 = x0
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / np.maximum(g, eps), 0.0, None)))
        return np.column_stack([f1, f2])
    if base == "FDA2":
        y = decode_unit(tail, -1.0, 1.0)
        split = max(1, y.shape[1] // 2)
        h_t = max(0.05, 0.75 + 0.75 * v)
        g = 1.0 + np.sum(y[:, :split] ** 2, axis=1)
        exponent = 1.0 / np.maximum(h_t + np.mean((y[:, split:] - h_t) ** 2, axis=1), eps)
        f1 = x0
        f2 = g * (1.0 - np.power(np.clip(f1 / np.maximum(g, eps), 0.0, None), exponent))
        return np.column_stack([f1, f2])
    if base == "FDA3":
        y = decode_unit(tail, -1.0, 1.0)
        g_signal = abs(v)
        f_exp = 10.0 ** (2.0 * v)
        g = 1.0 + g_signal + np.sum((y - g_signal) ** 2, axis=1)
        f1 = np.power(x0, f_exp)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / np.maximum(g, eps), 0.0, None)))
        return np.column_stack([f1, f2])

    if base == "dMOP1":
        g = 1.0 + 9.0 * np.mean(tail ** 2, axis=1)
        h_t = 0.75 * v + 1.25
        f1 = x0
        f2 = g * (1.0 - np.power(np.clip(f1 / np.maximum(g, eps), 0.0, None), h_t))
        return np.column_stack([f1, f2])
    if base == "dMOP2":
        g_signal = abs(v)
        g = 1.0 + 9.0 * np.mean((tail - g_signal) ** 2, axis=1)
        h_t = 0.75 * v + 1.25
        f1 = x0
        f2 = g * (1.0 - np.power(np.clip(f1 / np.maximum(g, eps), 0.0, None), h_t))
        return np.column_stack([f1, f2])
    if base == "dMOP3":
        g_signal = abs(v)
        n = X.shape[1]
        r = min(n - 1, int(math.floor((n - 1) * g_signal)))
        f1 = np.clip(X[:, r], 0.0, 1.0)
        mask = [i for i in range(n) if i != r]
        g = 1.0 + 9.0 * np.mean((X[:, mask] - g_signal) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / np.maximum(g, eps), 0.0, None)))
        return np.column_stack([f1, f2])

    if base == "DF1":
        g_signal = abs(v)
        h_t = 0.75 * v + 1.25
        g = 1.0 + np.sum((tail - g_signal) ** 2, axis=1)
        f1 = x0
        f2 = g * (1.0 - np.power(np.clip(f1 / np.maximum(g, eps), 0.0, None), h_t))
        return np.column_stack([f1, f2])
    if base == "DF2":
        g_signal = abs(v)
        n = X.shape[1]
        r = min(n - 1, int(math.floor((n - 1) * g_signal)))
        f1 = np.clip(X[:, r], 0.0, 1.0)
        mask = [i for i in range(n) if i != r]
        g = 1.0 + np.sum((X[:, mask] - g_signal) ** 2, axis=1)
        f2 = g * (1.0 - np.sqrt(np.clip(f1 / np.maximum(g, eps), 0.0, None)))
        return np.column_stack([f1, f2])
    if base == "DF3":
        y = decode_unit(tail, -1.0, 2.0)
        h_t = v + 1.5
        g = 1.0 + np.sum((y - v - x0[:, None] ** h_t) ** 2, axis=1)
        f1 = x0
        f2 = g * (1.0 - np.power(np.clip(x0 / np.maximum(g, eps), 0.0, None), h_t))
        return np.column_stack([f1, f2])
    if base == "DF4":
        z = decode_unit(X, -2.0, 2.0)
        a = v
        b = 1.0 + abs(math.cos(0.5 * math.pi * benchmark_time(scenario, state, states)))
        h_t = 1.5 + a
        c = max(abs(a), a + b, eps)
        g = np.ones(X.shape[0])
        for i in range(1, X.shape[1]):
            g += (z[:, i] - (a * (z[:, 0] / c) ** 2 / (i + 1))) ** 2
        f1 = g * np.abs(z[:, 0] - a) ** h_t
        f2 = g * np.abs(z[:, 0] - a - b) ** h_t
        return np.column_stack([f1, f2])
    if base == "DF5":
        y = decode_unit(tail, -1.0, 1.0)
        w = math.floor(10.0 * v)
        g = 1.0 + np.sum((y - v) ** 2, axis=1)
        ripple = 0.02 * np.sin(w * math.pi * x0)
        f1 = g * (x0 + ripple)
        f2 = g * (1.0 - x0 + ripple)
        return np.column_stack([f1, f2])

    raise ValueError(f"Unknown scenario: {scenario}")

@lru_cache(maxsize=None)
def reference_front(scenario: str, state: int, states: int, n: int = 300) -> np.ndarray:
    tau = env_time(state, states)
    base = scenario_base(scenario)
    v = dynamic_signal(scenario, state, states)
    grid = np.linspace(0, 1, n)

    if base == "FDA1":
        return np.column_stack([grid, 1.0 - np.sqrt(grid)])
    if base == "FDA2":
        h_t = max(0.05, 0.75 + 0.75 * v)
        return np.column_stack([grid, 1.0 - np.power(grid, 1.0 / h_t)])
    if base == "FDA3":
        g_signal = abs(v)
        return np.column_stack([grid, (1.0 + g_signal) * (1.0 - np.sqrt(grid / (1.0 + g_signal)))])
    if base in {"dMOP1", "dMOP2", "DF1"}:
        h_t = 0.75 * v + 1.25
        return np.column_stack([grid, 1.0 - np.power(grid, h_t)])
    if base in {"dMOP3", "DF2"}:
        return np.column_stack([grid, 1.0 - np.sqrt(grid)])
    if base == "DF3":
        h_t = v + 1.5
        return np.column_stack([grid, 1.0 - np.power(grid, h_t)])
    if base == "DF4":
        a = v
        b = 1.0 + abs(math.cos(0.5 * math.pi * benchmark_time(scenario, state, states)))
        h_t = 1.5 + a
        x = np.linspace(a, a + b, n)
        return np.column_stack([np.abs(x - a) ** h_t, np.abs(x - a - b) ** h_t])
    if base == "DF5":
        w = math.floor(10.0 * v)
        ripple = 0.02 * np.sin(w * math.pi * grid)
        pf = np.column_stack([grid + ripple, 1.0 - grid + ripple])
        return pf[front_ranks_2d(pf) == 0]

    xs = np.zeros((n, 10))
    xs[:, 0] = grid
    sh = app_shift(base, tau)
    xs[:, 1:] = sh
    if base == "APP-SCHEDULING":
        idx = np.arange(1, 10)
        xs[:, 1:] = sh + 0.10 * np.sin(idx * math.pi * tau)
    if base == "APP-RESOURCE":
        xs[:, 1:] = np.linspace(sh - 0.08, sh + 0.08, 9)
    return eval_objectives(xs, scenario, state, states, count=False)

def front_ranks_2d(F: np.ndarray) -> np.ndarray:
    # Efficient non-dominated layer assignment for two minimization objectives.
    order = np.lexsort((F[:, 1], F[:, 0]))
    ranks = np.zeros(len(F), dtype=int)
    last_f2: List[float] = []
    for idx in order:
        y = F[idx, 1]
        placed = False
        for r in range(len(last_f2)):
            if y < last_f2[r] - 1e-12:
                ranks[idx] = r
                last_f2[r] = y
                placed = True
                break
        if not placed:
            ranks[idx] = len(last_f2)
            last_f2.append(y)
    return ranks

def crowding_distance(F: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    n = len(F)
    cd = np.zeros(n)
    for r in np.unique(ranks):
        idx = np.where(ranks == r)[0]
        if len(idx) <= 2:
            cd[idx] = 1e6
            continue
        vals = F[idx]
        local = np.zeros(len(idx))
        for m in range(F.shape[1]):
            ordm = np.argsort(vals[:, m])
            local[ordm[0]] = local[ordm[-1]] = 1e6
            denom = vals[ordm[-1], m] - vals[ordm[0], m]
            if denom <= 1e-12:
                continue
            for k in range(1, len(idx) - 1):
                local[ordm[k]] += (vals[ordm[k + 1], m] - vals[ordm[k - 1], m]) / denom
        cd[idx] = local
    return cd

def environmental_select_indices(F: np.ndarray, N: int) -> np.ndarray:
    """Return indices selected by the NSGA-II environmental rule."""
    ranks = front_ranks_2d(F)
    cd = crowding_distance(F, ranks)
    # sort by rank asc, crowding desc
    order = np.lexsort((-cd, ranks))
    return order[:N]

def environmental_select(P: np.ndarray, F: np.ndarray, N: int) -> np.ndarray:
    return P[environmental_select_indices(F, N)]

@lru_cache(maxsize=None)
def metric_reference_point(scenario: str, states: int) -> Tuple[float, float]:
    """Scenario-level HV reference point computed from all true reference fronts."""
    fronts = [reference_front(scenario, s, states, n=500) for s in range(states)]
    vals = np.vstack(fronts)
    hi = np.nanmax(vals, axis=0)
    span = np.maximum(hi - np.nanmin(vals, axis=0), 1.0)
    ref = hi + 0.2 * span
    ref = np.maximum(ref, np.array([1.2, 1.2]))
    return float(ref[0]), float(ref[1])

def robust_objectives(P: np.ndarray, scenario: str, state: int, cfg: Config, penalty: float = 0.25) -> np.ndarray:
    """Robust dynamic surrogate objectives over a small temporal neighborhood.

    This inexpensive robust baseline approximates robustness to environmental
    uncertainty by selecting on the mean objective vector plus a dispersion
    penalty over adjacent states. It is used only for the RDMOEA-NSGA-II
    comparison; reported metrics are still evaluated at the true current state.
    """
    states = sorted({max(0, state - 1), state, min(cfg.states - 1, state + 1)})
    vals = np.stack([eval_objectives(P, scenario, s, cfg.states) for s in states], axis=0)
    return vals.mean(axis=0) + penalty * vals.std(axis=0)

def scalar_select(P: np.ndarray, F: np.ndarray, N: int, weights: np.ndarray) -> np.ndarray:
    # Fast decomposition-like selection. For each weight vector, the best scalarized
    # point is selected; remaining slots are filled by a normalized aggregate score.
    z = F.min(axis=0)
    Fp = F - z
    # matrix of Tchebycheff scores, shape (num_points, num_weights)
    scores = np.maximum(Fp[:, None, 0] * weights[None, :, 0], Fp[:, None, 1] * weights[None, :, 1])
    candidates = list(dict.fromkeys(np.argmin(scores, axis=0).tolist()))
    if len(candidates) >= N:
        return P[candidates[:N]]
    Fn = (F - F.min(axis=0)) / np.maximum(F.max(axis=0) - F.min(axis=0), 1e-12)
    agg = Fn[:, 0] + Fn[:, 1]
    fill_order = [i for i in np.argsort(agg).tolist() if i not in set(candidates)]
    idx = candidates + fill_order[:max(0, N - len(candidates))]
    return P[idx[:N]]

def make_offspring(P: np.ndarray, rng: np.random.Generator, cfg: Config, sigma: float | None = None) -> np.ndarray:
    N, D = P.shape
    M = N * cfg.offspring_multiplier
    a = P[rng.integers(0, N, size=M)]
    b = P[rng.integers(0, N, size=M)]
    blend = rng.random((M, 1))
    child = blend * a + (1 - blend) * b
    # occasional direct copy to preserve parents
    mask_cross = rng.random(M) < cfg.crossover_rate
    child[~mask_cross] = a[~mask_cross]
    sig = cfg.mutation_sigma if sigma is None else sigma
    mut = rng.normal(0, sig, size=child.shape)
    pm = 1.0 / D
    child += mut * (rng.random(child.shape) < pm)
    return np.clip(child, 0.0, 1.0)

def nondom_archive(P: np.ndarray, scenario: str, state: int, cfg: Config, max_size: int) -> np.ndarray:
    F = eval_objectives(P, scenario, state, cfg.states)
    ranks = front_ranks_2d(F)
    nd = P[ranks == 0]
    if len(nd) == 0:
        return environmental_select(P, F, min(max_size, len(P)))
    if len(nd) <= max_size:
        return nd.copy()
    Fnd = eval_objectives(nd, scenario, state, cfg.states)
    return environmental_select(nd, Fnd, max_size)

def predict_candidates(memory_hist: List[np.ndarray], N: int, D: int, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Generate prediction candidates from recent archive centroids.

    cfg.prediction_window is a centroid-history length. Therefore, w recent
    centroids provide at most w-1 consecutive displacement vectors. The formula
    in the manuscript uses the same convention.
    """
    if len(memory_hist) < 2 or len(memory_hist[-1]) == 0:
        return rng.random((N, D))
    centroids = [m.mean(axis=0) for m in memory_hist[-cfg.prediction_window:] if len(m) > 0]
    if len(centroids) < 2:
        base = memory_hist[-1]
        picked = base[rng.integers(0, len(base), size=N)]
        return np.clip(picked + rng.normal(0, cfg.mutation_sigma, size=picked.shape), 0, 1)
    movement = np.zeros(D)
    weights = np.linspace(0.5, 1.0, len(centroids) - 1)
    weights = weights / weights.sum()
    for w, a, b in zip(weights, centroids[:-1], centroids[1:]):
        movement += w * (b - a)
    base = memory_hist[-1]
    picked = base[rng.integers(0, len(base), size=N)]
    return np.clip(picked + cfg.prediction_scale * movement + rng.normal(0, cfg.mutation_sigma * 0.7, size=picked.shape), 0, 1)

def transfer_candidates(memory_hist: List[np.ndarray], N: int, D: int, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    valid = [m for m in memory_hist if len(m) > 0]
    if not valid:
        return rng.random((N, D))
    # Bias transfer toward recent archives but keep older environments available.
    weights = np.linspace(0.35, 1.0, len(valid))
    weights = weights / weights.sum()
    counts = rng.multinomial(N, weights)
    chunks = []
    for archive, k in zip(valid, counts):
        if k <= 0:
            continue
        picked = archive[rng.integers(0, len(archive), size=k)]
        chunks.append(picked + rng.normal(0, cfg.mutation_sigma * 0.5, size=picked.shape))
    if not chunks:
        return rng.random((N, D))
    return np.clip(np.vstack(chunks), 0.0, 1.0)

def population_linear_prediction(pop_hist: List[np.ndarray], N: int, D: int, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    """Baseline linear prediction using the same centroid-history convention."""
    valid = [p for p in pop_hist[-cfg.prediction_window:] if len(p) > 0]
    if len(valid) < 2:
        return rng.random((N, D))
    centroids = [p.mean(axis=0) for p in valid]
    movement = centroids[-1] - centroids[-2]
    if len(centroids) > 2:
        movement = 0.65 * movement + 0.35 * (centroids[-2] - centroids[-3])
    base = valid[-1]
    picked = base[rng.integers(0, len(base), size=N)]
    return np.clip(picked + cfg.prediction_scale * movement + rng.normal(0, cfg.mutation_sigma * 0.55, size=picked.shape), 0, 1)

def make_prr_source_diagnostics(
    proposed_counts: Dict[str, int],
    accepted_labels: np.ndarray,
    *,
    population_size: int,
    pool_target: int,
    filter_candidates: bool,
) -> Dict[str, float]:
    """Summarize which PRR proposal sources survived environmental selection.

    The returned fields distinguish generated candidates from admitted candidates.
    Fractions are measured outcomes, not quotas: acceptance_fraction_s is
    accepted_s / proposed_s, while population_fraction_s is accepted_s / N.
    """
    sources = ["elite", "memory", "prediction", "diversity"]
    out: Dict[str, float] = {
        "pool_target": int(pool_target),
        "proposed_total": int(sum(int(proposed_counts.get(src, 0)) for src in sources)),
        "accepted_total": int(len(accepted_labels)),
        "filter_candidates": bool(filter_candidates),
    }
    for src in sources:
        proposed = int(proposed_counts.get(src, 0))
        accepted = int(np.sum(accepted_labels == src))
        out[f"proposed_{src}"] = proposed
        out[f"accepted_{src}"] = accepted
        out[f"acceptance_fraction_{src}"] = float(accepted / proposed) if proposed > 0 else 0.0
        out[f"population_fraction_{src}"] = float(accepted / population_size) if population_size > 0 else 0.0
    fallback = int(np.sum(accepted_labels == "fallback_current"))
    out["accepted_fallback_current"] = fallback
    out["population_fraction_fallback_current"] = float(fallback / population_size) if population_size > 0 else 0.0
    return out

def prr_response(
    P: np.ndarray,
    memory: np.ndarray,
    memory_hist: List[np.ndarray],
    scenario: str,
    state: int,
    cfg: Config,
    rng: np.random.Generator,
    *,
    use_elite: bool = True,
    use_memory: bool = True,
    use_prediction: bool = True,
    use_diversity: bool = True,
    filter_candidates: bool = True,
    return_diagnostics: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """Build an over-generated PRR redeployment pool with switchable components.

    The filtered PRR design intentionally generates more response candidates than
    the population size. The source ratios determine candidate-generation effort,
    not reserved admission slots. All generated candidates are re-evaluated in the
    current environment and only the best N candidates under environmental
    selection enter the redeployed population. When return_diagnostics=True,
    the routine also returns source-level proposal and acceptance counts. The
    direct-insertion ablation keeps a fixed number of predicted candidates before
    filling the remainder from the non-predicted pool; this isolates the effect
    of forced prediction admission.
    """
    N, D = cfg.pop_size, cfg.variables
    shares = {
        "elite": cfg.prr_elite_ratio if use_elite else 0.0,
        "memory": cfg.prr_memory_ratio if use_memory else 0.0,
        "prediction": cfg.prr_prediction_ratio if use_prediction else 0.0,
        "diversity": cfg.prr_diversity_ratio if use_diversity else 0.0,
    }
    total = sum(shares.values())
    if total <= 0:
        shares["elite"] = 1.0
        total = 1.0
    pool_target = max(N, int(round(N * max(cfg.prr_pool_multiplier, 1.0))))
    counts = {k: int(round(pool_target * v / total)) for k, v in shares.items()}
    diff = pool_target - sum(counts.values())
    counts["elite" if use_elite else next(k for k, v in counts.items() if v > 0)] += diff

    F = eval_objectives(P, scenario, state, cfg.states)
    parts: List[np.ndarray] = []
    labels: List[str] = []

    def add(block: np.ndarray, label: str) -> None:
        if len(block) > 0:
            parts.append(block)
            labels.extend([label] * len(block))

    if counts["elite"] > 0:
        k = min(counts["elite"], len(P))
        elite = environmental_select(P, F, k)
        if counts["elite"] > k:
            extra = elite[rng.integers(0, len(elite), size=counts["elite"] - k)]
            elite = np.vstack([elite, extra])
        add(elite, "elite")
    if counts["memory"] > 0:
        if len(memory) > 0:
            split = min(max(cfg.prr_recent_memory_split, 0.0), 1.0)
            k_recent = int(math.ceil(split * counts["memory"]))
            k_transfer = counts["memory"] - k_recent
            recent = memory[rng.integers(0, len(memory), size=k_recent)]
            if k_transfer > 0:
                transfer = transfer_candidates(memory_hist, k_transfer, D, rng, cfg)
                mem_block = np.vstack([recent, transfer])
            else:
                mem_block = recent
            add(mem_block, "memory")
        else:
            add(rng.random((counts["memory"], D)), "memory")
    if counts["prediction"] > 0:
        add(predict_candidates(memory_hist, counts["prediction"], D, rng, cfg), "prediction")
    if counts["diversity"] > 0:
        add(rng.random((counts["diversity"], D)), "diversity")

    pool = np.vstack(parts)
    labels_arr = np.asarray(labels)
    if not filter_candidates:
        # Direct insertion ablation: reserve a fixed prediction quota first, then
        # fill the remainder from non-predicted candidates under current-state selection.
        q_pred = min(int(round(N * shares.get("prediction", 0.0) / total)), int(np.sum(labels_arr == "prediction")))
        pred_idx = np.where(labels_arr == "prediction")[0]
        forced = pool[pred_idx[:q_pred]] if q_pred > 0 else np.empty((0, D))
        rest_idx = np.where(labels_arr != "prediction")[0]
        rest = pool[rest_idx]
        accepted_label_parts: List[np.ndarray] = []
        if q_pred > 0:
            accepted_label_parts.append(labels_arr[pred_idx[:q_pred]])
        if len(rest) > 0 and len(forced) < N:
            Frest = eval_objectives(rest, scenario, state, cfg.states)
            fill_local_idx = environmental_select_indices(Frest, min(N - len(forced), len(rest)))
            fill = rest[fill_local_idx]
            accepted_label_parts.append(labels_arr[rest_idx[fill_local_idx]])
            out = np.vstack([forced, fill]) if len(forced) else fill
        else:
            out = forced
        if len(out) < N:
            fallback_n = N - len(out)
            fill = P[rng.integers(0, len(P), size=fallback_n)]
            accepted_label_parts.append(np.asarray(["fallback_current"] * fallback_n, dtype=object))
            out = np.vstack([out, fill]) if len(out) else fill
        out = out[:N]
        accepted_labels = np.concatenate(accepted_label_parts)[:len(out)] if accepted_label_parts else np.asarray([], dtype=object)
        if return_diagnostics:
            diag = make_prr_source_diagnostics(counts, accepted_labels, population_size=N, pool_target=pool_target, filter_candidates=False)
            return out, diag
        return out

    Fpool = eval_objectives(pool, scenario, state, cfg.states)
    selected_idx = environmental_select_indices(Fpool, N)
    out = pool[selected_idx]
    if return_diagnostics:
        diag = make_prr_source_diagnostics(counts, labels_arr[selected_idx], population_size=N, pool_target=pool_target, filter_candidates=True)
        return out, diag
    return out

def abr_response(
    P: np.ndarray,
    memory: np.ndarray,
    memory_hist: List[np.ndarray],
    scenario: str,
    state: int,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """Adaptive boosting response used by ABR-NSGA-II.

    This helper is shared by the main experiment and the recovery audit so that
    ABR is evaluated with the same current-state source-scoring rule in both
    places. Candidate source pools are scored by both their median non-dominated
    front rank and mean objective value; lower values are better.
    """
    N, D = cfg.pop_size, cfg.variables
    F = eval_objectives(P, scenario, state, cfg.states)
    elite = environmental_select(P, F, int(0.35 * N))
    candidates = {
        "memory": memory[rng.integers(0, len(memory), size=N)] if len(memory) > 0 else rng.random((N, D)),
        "prediction": predict_candidates(memory_hist, N, D, rng, cfg),
        "diversity": rng.random((N, D)),
    }
    scores = []
    for cand in candidates.values():
        Fc = eval_objectives(cand, scenario, state, cfg.states)
        ranks = front_ranks_2d(Fc)
        scores.append(1.0 / (1.0 + float(np.median(ranks)) + float(Fc.mean())))
    probs = np.array(scores) / np.sum(scores)
    rem = N - len(elite)
    counts = rng.multinomial(rem, probs)
    parts = [elite]
    for cand, k in zip(candidates.values(), counts):
        if k > 0:
            parts.append(cand[rng.integers(0, len(cand), size=k)])
    pool = np.vstack(parts)
    Fpool = eval_objectives(pool, scenario, state, cfg.states)
    return environmental_select(pool, Fpool, N)

def hv_2d_min(F: np.ndarray, ref: Tuple[float, float] = (1.6, 2.4)) -> float:
    ranks = front_ranks_2d(F)
    nd = F[ranks == 0]
    if len(nd) == 0:
        return 0.0
    nd = nd[(nd[:, 0] < ref[0]) & (nd[:, 1] < ref[1])]
    if len(nd) == 0:
        return 0.0
    order = np.argsort(nd[:, 0])
    pts = nd[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += max(ref[0] - x, 0) * max(prev_y - y, 0)
            prev_y = y
    return float(max(hv, 0.0))

def igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    ranks = front_ranks_2d(F)
    nd = F[ranks == 0]
    if len(nd) == 0:
        nd = F
    # vectorized distances in chunks
    total = 0.0
    for i in range(0, len(ref_front), 100):
        R = ref_front[i:i+100]
        dist = np.sqrt(((R[:, None, :] - nd[None, :, :]) ** 2).sum(axis=2))
        total += float(dist.min(axis=1).sum())
    return total / len(ref_front)

def compromise_solution(P: np.ndarray, F: np.ndarray) -> np.ndarray:
    lo = F.min(axis=0)
    hi = F.max(axis=0)
    Fn = (F - lo) / np.maximum(hi - lo, 1e-12)
    score = np.sqrt((Fn ** 2).sum(axis=1))
    return P[int(np.argmin(score))]

def run_one(scenario: str, algorithm: str, run: int, cfg: Config) -> Dict[str, float]:
    # Use the same stochastic stream for the complete PRR ablation baseline as
    # the main PRR-NSGA-II rows. This prevents seed-induced differences from
    # making an algorithmically identical PRR-complete row look like a separate
    # setting. Other ablation variants keep variant-specific streams.
    seed_algorithm = "PRR-NSGA-II" if algorithm == "PRR-complete" else algorithm
    rng = np.random.default_rng(stable_seed(scenario, seed_algorithm, run, cfg.pop_size, cfg.states, cfg.generations_per_state))
    N, D = cfg.pop_size, cfg.variables
    P = rng.random((N, D))
    memory = np.empty((0, D))
    memory_hist: List[np.ndarray] = []
    population_hist: List[np.ndarray] = []
    hvs, igds, comp = [], [], []
    weights = np.column_stack([np.linspace(0.05, 0.95, N), np.linspace(0.95, 0.05, N)])
    # Environmental states are intentionally zero-based: state = 0, ..., cfg.states - 1.
    # The first state has no change-response step; response mechanisms are applied for state > 0.
    for state in range(cfg.states):
        # response to environmental change before search
        if state > 0:
            if algorithm == "DNSGA-II":
                k = max(1, int(0.25 * N))
                F = eval_objectives(P, scenario, state, cfg.states)
                P = environmental_select(P, F, N - k)
                P = np.vstack([P, rng.random((k, D))])
            elif algorithm == "MNSGA-II":
                if len(memory) > 0:
                    k = min(len(memory), int(0.35 * N))
                    F = eval_objectives(P, scenario, state, cfg.states)
                    Pbase = environmental_select(P, F, N - k)
                    Pmem = memory[rng.integers(0, len(memory), size=k)]
                    P = np.vstack([Pbase, Pmem])
            elif algorithm == "PPS-NSGA-II":
                k = int(0.40 * N)
                pred = predict_candidates(memory_hist, k, D, rng, cfg)
                F = eval_objectives(P, scenario, state, cfg.states)
                Pbase = environmental_select(P, F, N - k)
                P = np.vstack([Pbase, pred])
            elif algorithm == "DMOEA/D":
                km, kp, kd = int(0.15 * N), int(0.15 * N), int(0.10 * N)
                F = eval_objectives(P, scenario, state, cfg.states)
                base = scalar_select(P, F, N - km - kp - kd, weights)
                mem = memory[rng.integers(0, len(memory), size=km)] if len(memory) > 0 else rng.random((km, D))
                pred = predict_candidates(memory_hist, kp, D, rng, cfg)
                pool = np.vstack([base, mem, pred, rng.random((kd, D))])
                Fpool = eval_objectives(pool, scenario, state, cfg.states)
                P = scalar_select(pool, Fpool, N, weights)
            elif algorithm == "ARVLP-NSGA-II":
                kp, kd = int(0.40 * N), int(0.10 * N)
                F = eval_objectives(P, scenario, state, cfg.states)
                base = environmental_select(P, F, N - kp - kd)
                pred = population_linear_prediction(population_hist, kp, D, rng, cfg)
                pool = np.vstack([base, pred, rng.random((kd, D))])
                Fpool = eval_objectives(pool, scenario, state, cfg.states)
                P = environmental_select(pool, Fpool, N)
            elif algorithm == "KTR-NSGA-II":
                kt, kp = int(0.35 * N), int(0.20 * N)
                F = eval_objectives(P, scenario, state, cfg.states)
                base = environmental_select(P, F, N - kt - kp)
                transfer = transfer_candidates(memory_hist, kt, D, rng, cfg)
                pred = predict_candidates(memory_hist, kp, D, rng, cfg)
                pool = np.vstack([base, transfer, pred])
                Fpool = eval_objectives(pool, scenario, state, cfg.states)
                P = environmental_select(pool, Fpool, N)
            elif algorithm == "ABR-NSGA-II":
                P = abr_response(P, memory, memory_hist, scenario, state, cfg, rng)
            elif algorithm == "RDMOEA-NSGA-II":
                # Robust dynamic baseline: select against a temporal neighborhood
                # to reduce sensitivity to short-lived environmental fluctuations.
                km, kd = int(0.25 * N), int(0.10 * N)
                Frob = robust_objectives(P, scenario, state, cfg)
                base = environmental_select(P, Frob, N - km - kd)
                mem = memory[rng.integers(0, len(memory), size=km)] if len(memory) > 0 else rng.random((km, D))
                div = rng.random((kd, D))
                pool = np.vstack([base, mem, div])
                Fpool = robust_objectives(pool, scenario, state, cfg)
                P = environmental_select(pool, Fpool, N)
            elif algorithm in {"PRR-NSGA-II", "PRR-complete"}:
                P = prr_response(P, memory, memory_hist, scenario, state, cfg, rng)
            elif algorithm == "PRR-no-prediction":
                P = prr_response(P, memory, memory_hist, scenario, state, cfg, rng, use_prediction=False)
            elif algorithm == "PRR-no-memory":
                P = prr_response(P, memory, memory_hist, scenario, state, cfg, rng, use_memory=False)
            elif algorithm == "PRR-no-diversity":
                P = prr_response(P, memory, memory_hist, scenario, state, cfg, rng, use_diversity=False)
            elif algorithm in {"PRR-direct-insertion", "PRR-no-selection-filter"}:
                P = prr_response(P, memory, memory_hist, scenario, state, cfg, rng, filter_candidates=False)
            elif algorithm == "PRR-no-elite":
                P = prr_response(P, memory, memory_hist, scenario, state, cfg, rng, use_elite=False)
        # evolutionary search in current state
        for _ in range(cfg.generations_per_state):
            if algorithm == "DMOEA/D":
                child = make_offspring(P, rng, cfg, sigma=cfg.mutation_sigma * 0.85)
                pool = np.vstack([P, child])
                Fpool = eval_objectives(pool, scenario, state, cfg.states)
                P = scalar_select(pool, Fpool, N, weights)
            elif algorithm == "RDMOEA-NSGA-II":
                child = make_offspring(P, rng, cfg, sigma=cfg.mutation_sigma * 0.75)
                pool = np.vstack([P, child])
                Fpool = robust_objectives(pool, scenario, state, cfg)
                P = environmental_select(pool, Fpool, N)
            else:
                child = make_offspring(P, rng, cfg)
                pool = np.vstack([P, child])
                Fpool = eval_objectives(pool, scenario, state, cfg.states)
                P = environmental_select(pool, Fpool, N)
        F = eval_objectives(P, scenario, state, cfg.states)
        ref = reference_front(scenario, state, cfg.states, n=300)
        ref_point = metric_reference_point(scenario, cfg.states)
        ref_area = max(ref_point[0] * ref_point[1], 1e-12)
        hvs.append(hv_2d_min(F, ref=ref_point) / ref_area)
        igds.append(igd(F, ref))
        comp.append(compromise_solution(P, F))
        # memory update
        nd = nondom_archive(P, scenario, state, cfg, cfg.memory_size)
        if len(memory) == 0:
            memory = nd
        else:
            memory = np.vstack([memory, nd])
            Fmem = eval_objectives(memory, scenario, state, cfg.states)
            memory = environmental_select(memory, Fmem, min(cfg.memory_size, len(memory)))
        memory_hist.append(nd)
        population_hist.append(P.copy())
    hvs = np.array(hvs)
    igds = np.array(igds)
    # stability normalized by reference hv range
    hv_diffs = np.abs(np.diff(hvs))
    stability = float(1.0 / (1.0 + hv_diffs.mean())) if len(hv_diffs) else 1.0
    comp = np.array(comp)
    drift = float(np.linalg.norm(np.diff(comp, axis=0), axis=1).mean()) if len(comp) > 1 else 0.0
    qdr = float(hvs.mean() / (1.0 + drift))
    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "run": run,
        "HV": float(hvs.mean()),
        "IGD": float(igds.mean()),
        "Stability": stability,
        "DecisionDrift": drift,
        "QDR": qdr,
    }

def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby(["scenario", "algorithm"]).agg(
        HV_mean=("HV", "mean"), HV_std=("HV", "std"),
        IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"),
        Stability_mean=("Stability", "mean"), Stability_std=("Stability", "std"),
        DecisionDrift_mean=("DecisionDrift", "mean"), DecisionDrift_std=("DecisionDrift", "std"),
        QDR_mean=("QDR", "mean"), QDR_std=("QDR", "std"),
    ).reset_index()

def overall_summary(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby("algorithm").agg(
        HV_mean=("HV", "mean"), HV_std=("HV", "std"),
        IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"),
        Stability_mean=("Stability", "mean"), Stability_std=("Stability", "std"),
        DecisionDrift_mean=("DecisionDrift", "mean"), DecisionDrift_std=("DecisionDrift", "std"),
        QDR_mean=("QDR", "mean"), QDR_std=("QDR", "std"),
    ).reset_index().sort_values("HV_mean", ascending=False)

def average_ranks(raw: pd.DataFrame, algorithms: List[str] | None = None) -> pd.DataFrame:
    """Average ranks over scenario-level means.

    Earlier versions ranked scenario-run blocks. The stored main runs use
    deterministic algorithm-specific random streams, so run identifiers are not
    treated as paired blocks across algorithms. Scenario-level ranking prevents
    scale-heavy scenarios from dominating the aggregation and avoids artificial
    pairing.
    """
    algorithms = algorithms or ALGORITHMS
    rows = []
    means = raw.groupby(["scenario", "algorithm"]).mean(numeric_only=True).reset_index()
    for metric in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]:
        wide = means.pivot_table(index="scenario", columns="algorithm", values=metric)
        wide = wide[[a for a in algorithms if a in wide.columns]].dropna()
        ascending = metric in ["IGD", "DecisionDrift"]
        ranks = wide.rank(axis=1, method="average", ascending=ascending)
        for alg, val in ranks.mean(axis=0).items():
            rows.append({"metric": metric, "algorithm": alg, "average_rank": float(val)})
    return pd.DataFrame(rows)

def win_tie_loss(raw: pd.DataFrame, target: str = "PRR-NSGA-II", algorithms: List[str] | None = None) -> pd.DataFrame:
    algorithms = algorithms or ALGORITHMS
    rows = []
    means = raw.groupby(["scenario", "algorithm"]).mean(numeric_only=True).reset_index()
    for metric in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]:
        lower_better = metric in ["IGD", "DecisionDrift"]
        for alg in algorithms:
            if alg == target:
                continue
            wins = ties = losses = 0
            for scenario in sorted(raw.scenario.unique()):
                sub = means[means.scenario == scenario]
                tv = sub[sub.algorithm == target][metric]
                ov = sub[sub.algorithm == alg][metric]
                if tv.empty or ov.empty:
                    continue
                diff = float(tv.iloc[0] - ov.iloc[0])
                tol = 1e-10
                if abs(diff) <= tol:
                    ties += 1
                elif (diff < 0 and lower_better) or (diff > 0 and not lower_better):
                    wins += 1
                else:
                    losses += 1
            rows.append({"metric": metric, "comparison": f"{target} vs {alg}", "wins": wins, "ties": ties, "losses": losses})
    return pd.DataFrame(rows)

def holm_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0
    for i, idx in enumerate(order):
        val = min((m - i) * pvals[idx], 1.0)
        prev = max(prev, val)
        adj[idx] = prev
    return adj.tolist()

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # P(x>y)-P(x<y), positive means x higher than y.
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return (gt - lt) / (len(x) * len(y))

def statistical_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Independent PRR-vs-baseline tests with Holm correction.

    Because main results are generated with algorithm-specific stochastic
    streams, sorted run indices are not a valid natural pairing. Pairwise tests
    therefore use two-sided Mann-Whitney U tests. Cliff's delta is reported with
    the sign convention that positive values favor PRR.
    """
    rows = []
    if mannwhitneyu is None:
        return pd.DataFrame()
    comparisons = [a for a in ALGORITHMS if a != "PRR-NSGA-II"]
    metrics = ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]
    for metric in metrics:
        pvals = []
        meta = []
        higher_better = metric not in ["IGD", "DecisionDrift"]
        for scenario in SCENARIOS:
            sub = raw[raw.scenario == scenario]
            prr = sub[sub.algorithm == "PRR-NSGA-II"][metric].values
            for alg in comparisons:
                other = sub[sub.algorithm == alg][metric].values
                if len(prr) == 0 or len(other) == 0:
                    continue
                try:
                    p = float(mannwhitneyu(prr, other, alternative="two-sided").pvalue)
                except ValueError:
                    p = 1.0
                if higher_better:
                    delta = cliffs_delta(prr, other)
                    better = np.mean(prr) > np.mean(other)
                else:
                    delta = cliffs_delta(-prr, -other)
                    better = np.mean(prr) < np.mean(other)
                pvals.append(p)
                meta.append((scenario, alg, metric, p, delta, better, float(np.mean(prr)), float(np.mean(other))))
        adj = holm_correction(pvals) if pvals else []
        for rec, padj in zip(meta, adj):
            scenario, alg, metric, p, delta, better, prr_mean, other_mean = rec
            rows.append({
                "scenario": scenario, "metric": metric, "comparison": f"PRR-NSGA-II vs {alg}",
                "PRR_mean": prr_mean, "other_mean": other_mean,
                "raw_p_mann_whitney": p, "holm_p": padj,
                "cliffs_delta_positive_favors_PRR": delta,
                "PRR_better_by_mean": better,
                "significant_0.05": padj < 0.05,
            })
    return pd.DataFrame(rows)

def friedman_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Friedman tests over scenario-level mean blocks."""
    rows = []
    if friedmanchisquare is None:
        return pd.DataFrame()
    means = raw.groupby(["scenario", "algorithm"]).mean(numeric_only=True).reset_index()
    for metric in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]:
        wide = means.pivot_table(index="scenario", columns="algorithm", values=metric)
        wide = wide[ALGORITHMS].dropna()
        arrays = [wide[a].values for a in ALGORITHMS]
        try:
            stat, p = friedmanchisquare(*arrays)
        except Exception:
            stat, p = np.nan, np.nan
        rows.append({"metric": metric, "blocking": "scenario-level mean", "friedman_statistic": stat, "p_value": p})
    return pd.DataFrame(rows)

def run_one_prr_source_diagnostics(scenario: str, algorithm: str, run: int, cfg: Config) -> List[Dict[str, float]]:
    """Run a PRR-family algorithm and return per-change source diagnostics.

    This uses the same state loop, random seed convention, response routine,
    and post-response evolutionary updates as run_one(), but records which
    proposal labels are admitted immediately after each PRR response.
    """
    if not algorithm.startswith("PRR"):
        raise ValueError("source diagnostics are only defined for PRR-family algorithms")
    seed_algorithm = "PRR-NSGA-II" if algorithm == "PRR-complete" else algorithm
    rng = np.random.default_rng(stable_seed(scenario, seed_algorithm, run, cfg.pop_size, cfg.states, cfg.generations_per_state))
    N, D = cfg.pop_size, cfg.variables
    P = rng.random((N, D))
    memory = np.empty((0, D))
    memory_hist: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []

    for state in range(cfg.states):
        if state > 0:
            kwargs = {}
            if algorithm == "PRR-no-prediction":
                kwargs["use_prediction"] = False
            elif algorithm == "PRR-no-memory":
                kwargs["use_memory"] = False
            elif algorithm == "PRR-no-diversity":
                kwargs["use_diversity"] = False
            elif algorithm in {"PRR-direct-insertion", "PRR-no-selection-filter"}:
                kwargs["filter_candidates"] = False
            elif algorithm == "PRR-no-elite":
                kwargs["use_elite"] = False
            P, diag = prr_response(P, memory, memory_hist, scenario, state, cfg, rng, return_diagnostics=True, **kwargs)
            rows.append({
                "scenario": scenario,
                "algorithm": algorithm,
                "run": run,
                "state": state,
                "response_index": state,
                **diag,
            })

        for _ in range(cfg.generations_per_state):
            child = make_offspring(P, rng, cfg)
            pool = np.vstack([P, child])
            Fpool = eval_objectives(pool, scenario, state, cfg.states)
            P = environmental_select(pool, Fpool, N)

        nd = nondom_archive(P, scenario, state, cfg, cfg.memory_size)
        if len(memory) == 0:
            memory = nd
        else:
            memory = np.vstack([memory, nd])
            Fmem = eval_objectives(memory, scenario, state, cfg.states)
            memory = environmental_select(memory, Fmem, min(cfg.memory_size, len(memory)))
        memory_hist.append(nd)
    return rows

def run_one_prr_source_diagnostics_star(args):
    return run_one_prr_source_diagnostics(*args)

def run_prr_source_diagnostic_tasks(tasks: List[Tuple[str, str, int, Config]], cfg: Config) -> List[Dict[str, float]]:
    if not tasks:
        return []
    workers = cfg.workers if cfg.workers and cfg.workers > 0 else min(cpu_count(), 8)
    if workers <= 1 or len(tasks) < 4:
        nested = [run_one_prr_source_diagnostics_star(t) for t in tasks]
    else:
        workers = min(workers, len(tasks))
        with Pool(processes=workers) as pool:
            nested = list(pool.imap_unordered(run_one_prr_source_diagnostics_star, tasks, chunksize=max(1, len(tasks) // (workers * 8))))
    return [row for rows in nested for row in rows]

def run_one_star(args):
    return run_one(*args)

def run_one_with_evaluation_count(scenario: str, algorithm: str, run: int, cfg: Config) -> Dict[str, float]:
    """Run one experiment while counting objective-vector evaluations."""
    with evaluation_accounting() as counter:
        metrics = run_one(scenario, algorithm, run, cfg)
    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "run": run,
        "objective_vector_evaluations": counter.objective_vector_evaluations,
        "objective_function_calls": counter.objective_function_calls,
        "HV": metrics["HV"],
        "IGD": metrics["IGD"],
        "Stability": metrics["Stability"],
        "DecisionDrift": metrics["DecisionDrift"],
        "QDR": metrics["QDR"],
    }


def run_one_evaluation_count_star(args):
    return run_one_with_evaluation_count(*args)


def run_evaluation_count_tasks(tasks: List[Tuple[str, str, int, Config]], cfg: Config) -> List[Dict[str, float]]:
    if not tasks:
        return []
    workers = cfg.workers if cfg.workers and cfg.workers > 0 else min(cpu_count(), 8)
    if workers <= 1 or len(tasks) < 4:
        return [run_one_evaluation_count_star(t) for t in tasks]
    workers = min(workers, len(tasks))
    with Pool(processes=workers) as pool:
        return list(pool.imap_unordered(run_one_evaluation_count_star, tasks, chunksize=max(1, len(tasks) // (workers * 8))))


def run_tasks(tasks: List[Tuple[str, str, int, Config]], cfg: Config) -> List[Dict[str, float]]:
    if not tasks:
        return []
    workers = cfg.workers if cfg.workers and cfg.workers > 0 else min(cpu_count(), 8)
    if workers <= 1 or len(tasks) < 4:
        return [run_one_star(t) for t in tasks]
    workers = min(workers, len(tasks))
    with Pool(processes=workers) as pool:
        return list(pool.imap_unordered(run_one_star, tasks, chunksize=max(1, len(tasks) // (workers * 8))))

def benchmark_catalog() -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        base = scenario_base(scenario)
        frequency, severity = scenario_profile(scenario)
        if base.startswith("FDA"):
            family = "FDA"
        elif base.startswith("dMOP"):
            family = "dMOP"
        elif base.startswith("DF"):
            family = "Adapted DF-style"
        else:
            family = "application-inspired"
        rows.append({
            "scenario": scenario,
            "base_problem": base,
            "family": family,
            "frequency_multiplier": frequency,
            "severity_multiplier": severity,
        })
    return pd.DataFrame(rows)

def run_main(cfg: Config, outdir: Path) -> None:
    tasks = [(scenario, alg, run, cfg) for scenario in SCENARIOS for alg in ALGORITHMS for run in range(cfg.runs)]
    rows = run_tasks(tasks, cfg)
    raw = pd.DataFrame(rows)
    chunk_count = min(8, max(1, math.ceil(len(raw) / 500)))
    for idx, chunk_idx in enumerate(np.array_split(np.arange(len(raw)), chunk_count)):
        chunk = raw.iloc[chunk_idx]
        chunk.to_csv(outdir / f"raw_chunk_{idx:02d}.csv", index=False)
    raw.to_csv(outdir / "raw_run_metrics.csv", index=False)
    summarize(raw).to_csv(outdir / "summary_by_scenario_algorithm.csv", index=False)
    overall_summary(raw).to_csv(outdir / "overall_summary.csv", index=False)
    average_ranks(raw).to_csv(outdir / "average_ranks.csv", index=False)
    win_tie_loss(raw).to_csv(outdir / "win_tie_loss.csv", index=False)
    statistical_tests(raw).to_csv(outdir / "statistical_tests_prr_vs_baselines.csv", index=False)
    friedman_tests(raw).to_csv(outdir / "friedman_tests.csv", index=False)

def run_ablation(cfg: Config, outdir: Path) -> None:
    """Run the manuscript-facing all-scenario ablation.

    Earlier revisions used a five-scenario representative diagnostic and wrote
    generic files such as ablation_summary.csv. To avoid ambiguity, the current
    manuscript-facing ablation covers all 20 scenarios and writes only files with
    the ablation_all20_ prefix. Legacy five-scenario outputs, if retained for
    provenance, should be kept outside the primary results directory.
    """
    tasks = [(scenario, variant, run, cfg) for scenario in SCENARIOS for variant in ABLATION_ALGORITHMS for run in range(cfg.runs)]
    rows = [{**row, "variant": row["algorithm"]} for row in run_tasks(tasks, cfg)]
    df = pd.DataFrame(rows)
    df["algorithm"] = df["variant"]
    df = df.drop(columns=["variant"])
    df.to_csv(outdir / "ablation_all20_raw_metrics.csv", index=False)
    overall = df.groupby("algorithm").agg(
        HV_mean=("HV", "mean"), HV_std=("HV", "std"),
        IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"),
        Stability_mean=("Stability", "mean"), Stability_std=("Stability", "std"),
        DecisionDrift_mean=("DecisionDrift", "mean"), DecisionDrift_std=("DecisionDrift", "std"),
        QDR_mean=("QDR", "mean"), QDR_std=("QDR", "std"),
    ).reset_index()
    overall.to_csv(outdir / "ablation_all20_overall_summary.csv", index=False)
    scenario_summary = df.groupby(["scenario", "algorithm"]).agg(
        HV_mean=("HV", "mean"), HV_std=("HV", "std"),
        IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"),
        Stability_mean=("Stability", "mean"), Stability_std=("Stability", "std"),
        DecisionDrift_mean=("DecisionDrift", "mean"), DecisionDrift_std=("DecisionDrift", "std"),
        QDR_mean=("QDR", "mean"), QDR_std=("QDR", "std"),
    ).reset_index()
    scenario_summary.to_csv(outdir / "ablation_all20_by_scenario_summary.csv", index=False)
    average_ranks(df, ABLATION_ALGORITHMS).to_csv(outdir / "ablation_all20_average_ranks.csv", index=False)
    win_tie_loss(df, target="PRR-complete", algorithms=ABLATION_ALGORITHMS).to_csv(outdir / "ablation_all20_win_tie_loss.csv", index=False)
    winner_rows = []
    for metric in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]:
        higher_is_better = metric in ["HV", "Stability", "QDR"]
        for scenario in SCENARIOS:
            sub = scenario_summary[scenario_summary.scenario == scenario].copy()
            col = f"{metric}_mean"
            best = sub.sort_values(col, ascending=not higher_is_better).iloc[0]
            complete = sub[sub.algorithm == "PRR-complete"].iloc[0]
            gap = complete[col] - best[col] if higher_is_better else best[col] - complete[col]
            winner_rows.append({
                "scenario": scenario,
                "metric": metric,
                "best_variant": best["algorithm"],
                "best_mean": best[col],
                "PRR_complete_mean": complete[col],
                "complete_gap": gap,
            })
    pd.DataFrame(winner_rows).to_csv(outdir / "ablation_all20_scenario_metric_winners.csv", index=False)

def summarize_prr_source_diagnostics(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sources = ["elite", "memory", "prediction", "diversity"]
    long_rows = []
    for src in sources:
        tmp = raw[["scenario", "algorithm", "run", "state", f"proposed_{src}", f"accepted_{src}", f"acceptance_fraction_{src}", f"population_fraction_{src}"]].copy()
        tmp.columns = ["scenario", "algorithm", "run", "state", "proposed_count", "accepted_count", "acceptance_fraction", "population_fraction"]
        tmp["source"] = src
        long_rows.append(tmp)
    long = pd.concat(long_rows, ignore_index=True)
    summary_by_scenario = long.groupby(["scenario", "algorithm", "source"]).agg(
        response_events=("state", "count"),
        proposed_mean=("proposed_count", "mean"),
        accepted_mean=("accepted_count", "mean"),
        accepted_std=("accepted_count", "std"),
        acceptance_fraction_mean=("acceptance_fraction", "mean"),
        population_fraction_mean=("population_fraction", "mean"),
        accepted_sum=("accepted_count", "sum"),
        proposed_sum=("proposed_count", "sum"),
    ).reset_index()
    summary_by_scenario["pooled_acceptance_fraction"] = summary_by_scenario["accepted_sum"] / summary_by_scenario["proposed_sum"].replace(0, np.nan)
    summary_by_scenario["pooled_acceptance_fraction"] = summary_by_scenario["pooled_acceptance_fraction"].fillna(0.0)

    overall = long.groupby(["algorithm", "source"]).agg(
        response_events=("state", "count"),
        proposed_mean=("proposed_count", "mean"),
        accepted_mean=("accepted_count", "mean"),
        accepted_std=("accepted_count", "std"),
        acceptance_fraction_mean=("acceptance_fraction", "mean"),
        population_fraction_mean=("population_fraction", "mean"),
        accepted_sum=("accepted_count", "sum"),
        proposed_sum=("proposed_count", "sum"),
    ).reset_index()
    overall["pooled_acceptance_fraction"] = overall["accepted_sum"] / overall["proposed_sum"].replace(0, np.nan)
    overall["pooled_acceptance_fraction"] = overall["pooled_acceptance_fraction"].fillna(0.0)
    return summary_by_scenario, overall

def run_prr_source_diagnostics(cfg: Config, outdir: Path) -> None:
    """Generate accepted-source diagnostics for the manuscript PRR rows."""
    algorithms = ["PRR-NSGA-II"]
    tasks = [(scenario, alg, run, cfg) for scenario in SCENARIOS for alg in algorithms for run in range(cfg.runs)]
    raw = pd.DataFrame(run_prr_source_diagnostic_tasks(tasks, cfg))
    raw = raw.sort_values(["scenario", "algorithm", "run", "state"]).reset_index(drop=True)
    raw.to_csv(outdir / "prr_source_diagnostics_raw.csv", index=False)
    by_scenario, overall = summarize_prr_source_diagnostics(raw)
    by_scenario.to_csv(outdir / "prr_source_diagnostics_by_scenario.csv", index=False)
    overall.to_csv(outdir / "prr_source_diagnostics_overall.csv", index=False)
    metadata = {
        "purpose": "accepted-source diagnostics for selection-filtered PRR",
        "algorithms": algorithms,
        "scenarios": SCENARIOS,
        "runs_per_scenario_algorithm": cfg.runs,
        "states": cfg.states,
        "response_states_per_run": max(0, cfg.states - 1),
        "generations_per_state": cfg.generations_per_state,
        "population_size": cfg.pop_size,
        "pool_multiplier": cfg.prr_pool_multiplier,
        "source_shares": {
            "elite": cfg.prr_elite_ratio,
            "memory": cfg.prr_memory_ratio,
            "prediction": cfg.prr_prediction_ratio,
            "diversity": cfg.prr_diversity_ratio,
        },
        "definition": (
            "proposed_* counts are candidates generated before filtering; accepted_* counts are candidates whose "
            "source labels survive the PRR environmental selection step. acceptance_fraction_* is accepted/proposed; "
            "population_fraction_* is accepted/pop_size."
        ),
    }
    with open(outdir / "prr_source_diagnostics_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(outdir / "README_PRR_SOURCE_DIAGNOSTICS.txt", "w") as f:
        f.write(
            "PRR accepted-source diagnostics\n"
            "===============================\n\n"
            "These files are generated by prr_response(return_diagnostics=True). They record\n"
            "which generated source labels survive the current-state NSGA-II environmental\n"
            "selection gate after each environmental change.\n\n"
            "Run example from _supp_work/src:\n"
            "  python dynamic_prr_experiment.py --mode source-diagnostics --out-dir ../results --runs 30 --generations 3 --workers 4\n\n"
            "Files:\n"
            "- prr_source_diagnostics_raw.csv: per scenario/run/state accepted-source counts and fractions.\n"
            "- prr_source_diagnostics_by_scenario.csv: scenario-source averages.\n"
            "- prr_source_diagnostics_overall.csv: overall source averages.\n"
            "- prr_source_diagnostics_metadata.json: protocol and field definitions.\n"
        )

def run_sensitivity(cfg: Config, outdir: Path) -> None:
    rows = []
    base_scenarios = ["FDA1", "dMOP2", "DF1-HF"]
    # smaller repetitions for sensitivity; still deterministic and reported separately
    sens_runs = min(15, cfg.runs)
    tests = []
    for g in [3, 5, 10, 20]:
        c = Config(**cfg.__dict__); c.generations_per_state = g; tests.append(("generations_per_state", g, c))
    for p in [50, 75, 100, 125]:
        c = Config(**cfg.__dict__); c.pop_size = p; tests.append(("population_size", p, c))
    for w in [2, 4, 6, 8]:
        c = Config(**cfg.__dict__); c.prediction_window = w; tests.append(("prediction_window", w, c))
    for sc in [0.50, 0.85, 1.15]:
        c = Config(**cfg.__dict__); c.prediction_scale = sc; tests.append(("prediction_scale", sc, c))
    for param, value, c in tests:
        tasks = [(scenario, "PRR-NSGA-II", run, c) for scenario in base_scenarios for run in range(sens_runs)]
        rows.extend({**row, "parameter": param, "value": value} for row in run_tasks(tasks, c))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sensitivity_raw_metrics.csv", index=False)
    df.groupby(["parameter", "value"]).agg(
        HV_mean=("HV", "mean"), HV_std=("HV", "std"),
        IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"),
        Stability_mean=("Stability", "mean"), Stability_std=("Stability", "std"),
        DecisionDrift_mean=("DecisionDrift", "mean"), DecisionDrift_std=("DecisionDrift", "std"),
        QDR_mean=("QDR", "mean"), QDR_std=("QDR", "std"),
    ).reset_index().to_csv(outdir / "sensitivity_summary.csv", index=False)

def run_evaluation_count_audit(cfg: Config, outdir: Path, audit_runs: int = 5) -> None:
    """Generate the manuscript evaluation-count audit from the live accounting hook.

    The audit is intentionally separate from performance scoring. It records the
    number of objective-vector evaluations triggered by the reference
    implementation under the same population size, number of states, and
    generations-per-state protocol.
    """
    audit_runs = max(1, int(audit_runs))
    c = Config(**cfg.__dict__)
    c.runs = audit_runs
    tasks = [(scenario, alg, run, c) for scenario in SCENARIOS for alg in ALGORITHMS for run in range(audit_runs)]
    raw = pd.DataFrame(run_evaluation_count_tasks(tasks, c))
    raw = raw.sort_values(["scenario", "algorithm", "run"]).reset_index(drop=True)
    raw.to_csv(outdir / "evaluation_count_raw.csv", index=False)

    audit = raw.groupby(["scenario", "algorithm"]).agg(
        counted_runs=("run", "nunique"),
        objective_vector_evaluations_per_run=("objective_vector_evaluations", "mean"),
        objective_vector_evaluations_min=("objective_vector_evaluations", "min"),
        objective_vector_evaluations_max=("objective_vector_evaluations", "max"),
        objective_function_calls_per_run=("objective_function_calls", "mean"),
        objective_function_calls_min=("objective_function_calls", "min"),
        objective_function_calls_max=("objective_function_calls", "max"),
    ).reset_index()
    audit.to_csv(outdir / "evaluation_count_audit.csv", index=False)

    summary = raw.groupby("algorithm").agg(
        counted_scenario_runs=("run", "count"),
        mean_FE=("objective_vector_evaluations", "mean"),
        min_FE=("objective_vector_evaluations", "min"),
        max_FE=("objective_vector_evaluations", "max"),
        mean_calls=("objective_function_calls", "mean"),
        min_calls=("objective_function_calls", "min"),
        max_calls=("objective_function_calls", "max"),
    ).reset_index().sort_values("mean_FE")
    summary.to_csv(outdir / "evaluation_count_summary.csv", index=False)

    metadata = {
        "purpose": "objective-vector evaluation accounting audit generated by eval_objectives accounting hooks",
        "scenarios": SCENARIOS,
        "algorithms": ALGORITHMS,
        "audit_runs_per_scenario_algorithm": audit_runs,
        "states": c.states,
        "generations_per_state": c.generations_per_state,
        "population_size": c.pop_size,
        "variables": c.variables,
        "counting_rule": (
            "Each active eval_objectives invocation counts as one objective_function_call; "
            "each candidate row in that invocation counts as one objective_vector_evaluation. "
            "Reference-front construction calls eval_objectives(..., count=False) and is excluded."
        ),
    }
    with open(outdir / "evaluation_count_audit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(outdir / "README_EVALUATION_COUNT_AUDIT.txt", "w") as f:
        f.write(
            "Evaluation-count audit\n"
            "======================\n\n"
            "These files are generated by the accounting hook in eval_objectives().\n"
            "Run example from _supp_work/src:\n"
            "  python dynamic_prr_experiment.py --mode evaluation-count --out-dir ../results --runs 30 --generations 3 --evaluation-audit-runs 5 --workers 4\n\n"
            "Files:\n"
            "- evaluation_count_raw.csv: per scenario/algorithm/run counts.\n"
            "- evaluation_count_audit.csv: scenario-algorithm averages and ranges.\n"
            "- evaluation_count_summary.csv: algorithm-level summary.\n"
            "- evaluation_count_audit_metadata.json: protocol and counting rule.\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--variables", type=int, default=10)
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--generations", type=int, default=3, help="Generations per environmental state; default matches the manuscript protocol.")
    ap.add_argument("--mode", choices=["main", "ablation", "sensitivity", "evaluation-count", "source-diagnostics", "all"], default="all")
    ap.add_argument("--workers", type=int, default=0, help="Parallel worker processes; 0 uses min(cpu_count, 8).")
    ap.add_argument("--evaluation-audit-runs", type=int, default=5, help="Runs per scenario-algorithm pair for evaluation-count audit; default reproduces the supplied audit CSVs.")
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config(runs=args.runs, pop_size=args.pop_size, variables=args.variables, states=args.states, generations_per_state=args.generations, output_dir=str(outdir), mode=args.mode, workers=args.workers)
    with open(outdir / "experiment_config.json", "w") as f:
        config = {**cfg.__dict__, "algorithms": ALGORITHMS, "scenarios": SCENARIOS}
        json.dump(config, f, indent=2)
    benchmark_catalog().to_csv(outdir / "benchmark_catalog.csv", index=False)
    if args.mode in ("main", "all"):
        run_main(cfg, outdir)
    if args.mode in ("ablation", "all"):
        run_ablation(cfg, outdir)
    if args.mode in ("sensitivity", "all"):
        run_sensitivity(cfg, outdir)
    if args.mode in ("evaluation-count", "all"):
        run_evaluation_count_audit(cfg, outdir, audit_runs=args.evaluation_audit_runs)
    if args.mode in ("source-diagnostics", "all"):
        run_prr_source_diagnostics(cfg, outdir)

if __name__ == "__main__":
    main()
