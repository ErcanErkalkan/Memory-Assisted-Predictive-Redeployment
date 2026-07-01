#!/usr/bin/env python3
"""Generate the manuscript generation-sensitivity table from sensitivity_summary.csv."""
import csv
from pathlib import Path

supp_root = Path(__file__).resolve().parents[1]
csv_path = supp_root / "results" / "sensitivity_summary.csv"

rows = []
with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["parameter"] == "generations_per_state":
            rows.append(row)
rows.sort(key=lambda row: float(row["value"]))

print(r"Generations & HV & IGD & Stability & QDR \\")
print(r"\midrule")
for row in rows:
    print(
        f"{int(float(row['value']))} & "
        f"{float(row['HV_mean']):.3f} & "
        f"{float(row['IGD_mean']):.3f} & "
        f"{float(row['Stability_mean']):.3f} & "
        f"{float(row['QDR_mean']):.3f} " + r"\\"
    )
