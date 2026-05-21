# PRR-NSGA-II Experiment Code

This repository contains only the experiment code for the PRR-NSGA-II study.

Manuscript files are intentionally omitted. This branch does not include the paper PDF, LaTeX sources, bibliography, generated figures, generated tables, raw result CSV files, cover letter, highlights, or submission documents.

## DOI

Archived experiment-code DOI: https://doi.org/10.5281/zenodo.20327988

## Included Code

- `_supp_work/src/dynamic_prr_experiment.py`: main dynamic multi-objective experiment implementation.
- `_supp_work/run_all20_ablation.py`: all-scenario ablation runner.
- `_supp_work/src/prr_quota_sensitivity.py`: source-quota sensitivity experiment.
- `_supp_work/src/prr_prediction_noise_stress.py`: prediction-noise stress experiment.
- `_supp_work/src/recovery_audit.py`: recovery audit experiment.
- `_supp_work/src/run_evaluation_count_audit.py`: objective-evaluation count audit.
- `_supp_work/src/run_prr_source_diagnostics.py`: PRR source proposal/acceptance diagnostics.
- `run_scalability_diagnostic.py`: scalability diagnostic runner.

## Zenodo

The repository includes `.zenodo.json` for Zenodo release metadata and `CITATION.cff` for GitHub citation metadata.

## Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## License

Code is released under the MIT License. See `LICENSE`.
