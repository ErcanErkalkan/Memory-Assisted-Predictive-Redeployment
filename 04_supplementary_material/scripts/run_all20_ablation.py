#!/usr/bin/env python3
"""Regenerate the manuscript-facing all-20-scenario ablation outputs.

Outputs are written to 04_supplementary_material/results with the ablation_all20_ prefix.
The default configuration matches the manuscript protocol: 30 runs, 20 states,
population size 100, 10 variables, and 3 generations per state.
PRR-complete uses the same stochastic stream as PRR-NSGA-II in the main
experiment, so its aggregate values are directly comparable to the main PRR row.
Other component-removal variants retain their variant-specific streams.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / "src" / "dynamic_prr_experiment.py").exists():
    SUPP_ROOT = ROOT
elif (ROOT / "04_supplementary_material" / "src" / "dynamic_prr_experiment.py").exists():
    SUPP_ROOT = ROOT / "04_supplementary_material"
else:
    raise FileNotFoundError("Could not locate 04_supplementary_material/src/dynamic_prr_experiment.py or src/dynamic_prr_experiment.py")
SRC = SUPP_ROOT / "src"
sys.path.insert(0, str(SRC))

from dynamic_prr_experiment import Config, run_ablation  # noqa: E402

MANUSCRIPT_GENERATIONS_PER_STATE = 3


def main() -> None:
    outdir = SUPP_ROOT / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        runs=30,
        pop_size=100,
        variables=10,
        states=20,
        generations_per_state=MANUSCRIPT_GENERATIONS_PER_STATE,
        workers=4,
        output_dir=str(outdir),
        mode="ablation",
    )
    assert cfg.generations_per_state == MANUSCRIPT_GENERATIONS_PER_STATE, (
        "All-20 ablation must use the manuscript protocol: "
        f"generations_per_state={MANUSCRIPT_GENERATIONS_PER_STATE}"
    )
    run_ablation(cfg, outdir)
    print(f"Wrote all-20-scenario ablation outputs to {outdir}")


if __name__ == "__main__":
    main()
