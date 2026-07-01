#!/usr/bin/env python3
"""Regenerate the manuscript quota-sensitivity LaTeX table from CSV results.

This keeps Table tab:quota-sensitivity synchronized with
results/quota_sensitivity/prr_quota_sensitivity_summary.csv.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

ORDER = [
    ("default_35_30_25_10_split55", "Default"),
    ("elite_low_20", "Low elite"),
    ("elite_high_50", "High elite"),
    ("memory_low_15", "Low memory"),
    ("memory_high_45", "High memory"),
    ("prediction_low_10", "Low prediction"),
    ("prediction_high_40", "High prediction"),
    ("diversity_low_02", "Low diversity"),
    ("diversity_high_25", "High diversity"),
    ("recent_memory_heavy_80", "Recent-heavy memory"),
    ("transfer_memory_heavy_25", "Transfer-heavy memory"),
]


def build_table(summary_csv: Path) -> str:
    df = pd.read_csv(summary_csv).set_index("variant")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Targeted 8-scenario, 12-run PRR source-quota sensitivity audit. The table reports mean HV, IGD, and QDR over 12 independent runs per representative scenario while varying the elite, memory, prediction, diversity, and recent-memory split parameters under the same selection-filtered admission rule. Lower IGD is better; higher HV and QDR are better.}",
        r"\label{tab:quota-sensitivity}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Variant & $q_e$ & $q_m$ & $q_p$ & $q_d$ & $r_m$ & HV & IGD & QDR \\",
        r"\midrule",
    ]
    for key, label in ORDER:
        if key not in df.index:
            continue
        row = df.loc[key]
        lines.append(
            f"{label} & {row['elite_ratio']:.2f} & {row['memory_ratio']:.2f} & "
            f"{row['prediction_ratio']:.2f} & {row['diversity_ratio']:.2f} & "
            f"{row['recent_memory_split']:.2f} & {row['HV_mean']:.3f} & "
            f"{row['IGD_mean']:.3f} & {row['QDR_mean']:.3f} " + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{0.5ex}",
        r"\footnotesize{Targeted diagnostic audit: 8 representative scenarios $\times$ 12 independent runs per scenario/variant. $r_m$ is the recent-memory share inside the memory/transfer component; $1-r_m$ is the historical-transfer share.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    supp_root = Path(__file__).resolve().parents[1]
    package_root = supp_root.parent
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=supp_root / "results" / "quota_sensitivity" / "prr_quota_sensitivity_summary.csv",
    )
    parser.add_argument("--output", type=Path, default=package_root / "02_latex_source" / "methodology_sensitivity_tables.tex")
    args = parser.parse_args()

    new_table = build_table(args.summary_csv)
    output = args.output
    if output.exists():
        text = output.read_text(encoding="utf-8")
        first = text.index(r"\begin{table}[htbp]")
        second = text.index(r"\begin{table}[htbp]", first + 1)
        text = new_table + text[second:]
    else:
        text = new_table
    output.write_text(text, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
