#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parents[1]
res = base / "results"
out = res / "manuscript_tables.tex"

alg_order = [
    "NSGA-II", "DNSGA-II", "MNSGA-II", "PPS-NSGA-II", "DMOEA/D",
    "ARVLP-NSGA-II", "KTR-NSGA-II", "ABR-NSGA-II", "RDMOEA-NSGA-II", "PRR-NSGA-II",
]

def mean_std(row, metric):
    return f"{float(row[f'{metric}_mean']):.3f} $\\pm$ {float(row[f'{metric}_std']):.3f}"

def best_mask(df, metric, lower=False):
    vals = df[f"{metric}_mean"].astype(float)
    best = vals.min() if lower else vals.max()
    return abs(vals - best) < 1e-12

def table_overall():
    raw = pd.read_csv(res / "raw_run_metrics.csv")
    scenario_count = raw["scenario"].nunique()
    run_count = raw["run"].nunique()
    df = pd.read_csv(res / "overall_summary.csv").set_index("algorithm").loc[alg_order].reset_index()
    masks = {m: best_mask(df, m, lower=m in ["IGD", "DecisionDrift"]) for m in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]}
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Overall mean $\\pm$ standard deviation across {scenario_count} scenarios and {run_count} runs.}}",
        "\\label{tab:overall}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Algorithm & HV $\\uparrow$ & IGD $\\downarrow$ & Stability $\\uparrow$ & Drift $\\downarrow$ & QDR $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for i, row in df.iterrows():
        cells = []
        for metric in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]:
            value = mean_std(row, metric)
            if masks[metric].iloc[i]:
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        lines.append(f"{row['algorithm']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)

def table_ranks():
    df = pd.read_csv(res / "average_ranks.csv")
    wide = df.pivot(index="algorithm", columns="metric", values="average_rank").loc[alg_order]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Scenario-level average ranks across scenario blocks. Lower rank is better.}",
        "\\label{tab:average-ranks}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Algorithm & HV & IGD & Stability & Drift & QDR \\\\",
        "\\midrule",
    ]
    for alg, row in wide.iterrows():
        lines.append(f"{alg} & {row['HV']:.2f} & {row['IGD']:.2f} & {row['Stability']:.2f} & {row['DecisionDrift']:.2f} & {row['QDR']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)

def table_wtl():
    df = pd.read_csv(res / "win_tie_loss.csv")
    keep = df[df.metric.isin(["HV", "IGD", "Stability", "QDR"])]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Scenario-level win/tie/loss counts for PRR-NSGA-II against baselines.}",
        "\\label{tab:wtl}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & HV & IGD & Stability & QDR \\\\",
        "\\midrule",
    ]
    for comp in keep.comparison.unique():
        sub = keep[keep.comparison == comp].set_index("metric")
        cells = [f"{int(sub.loc[m,'wins'])}/{int(sub.loc[m,'ties'])}/{int(sub.loc[m,'losses'])}" for m in ["HV", "IGD", "Stability", "QDR"]]
        lines.append(f"{comp.replace('PRR-NSGA-II vs ', 'vs ')} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)

def table_ablation():
    order = ["PRR-complete", "PRR-no-prediction", "PRR-no-memory", "PRR-no-diversity", "PRR-direct-insertion", "PRR-no-elite"]
    df = pd.read_csv(res / "ablation_all20_overall_summary.csv").set_index("algorithm").loc[order].reset_index()
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{All-scenario component-removal ablation and direct-insertion test across 20 scenarios and 30 runs under the three-generation-per-state manuscript protocol; PRR-complete is the main PRR-NSGA-II raw-run set renamed for ablation consistency.}",
        "\\label{tab:ablation}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Variant & HV $\\uparrow$ & IGD $\\downarrow$ & Stability $\\uparrow$ & Drift $\\downarrow$ & QDR $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [mean_std(row, m) for m in ["HV", "IGD", "Stability", "DecisionDrift", "QDR"]]
        lines.append(f"{row['algorithm']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)

def table_sensitivity_generations():
    df = pd.read_csv(res / "sensitivity_summary.csv")
    df = df[df.parameter == "generations_per_state"].sort_values("value")
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Sensitivity to generations per environmental state.}",
        "\\label{tab:sens-generations}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Generations & HV $\\uparrow$ & IGD $\\downarrow$ & Stability $\\uparrow$ & QDR $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [mean_std(row, m) for m in ["HV", "IGD", "Stability", "QDR"]]
        lines.append(f"{int(float(row['value']))} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)

out.write_text("\n".join([
    table_overall(),
    table_ranks(),
    table_wtl(),
    table_ablation(),
]), encoding="utf-8")
print(out)
