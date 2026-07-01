# Applied Soft Computing PRR-NSGA-II Final Package

This package contains the manuscript files prepared for submission to Applied Soft Computing.

## Structure

- `01_main_manuscript/` - final compiled manuscript PDF with shortened abstract, cleaned baseline labels, consolidated main-text tables, and added analytic admission/complexity notes.
- `02_latex_source/` - LaTeX source, BibTeX database, essential main-text table inputs, and figure files required to rebuild the manuscript.
- `03_submission_documents/` - cover letter, highlights, and graphical abstract.
- `04_supplementary_material/` - reproducibility scripts, stored results, extended diagnostic tables, figures, and supplementary README.

## Baseline-label note

Baseline labels are written as ARVLP-NSGA-II, KTR-NSGA-II, ABR-NSGA-II, and RDMOEA-NSGA-II. The baseline section uses neutral response-family terminology.

## Figure note

Two additional manuscript-facing figures were added as deterministic Matplotlib vector drawings:

- `prr_dynamic_setting.pdf`
- `prr_selection_filter_detail.pdf`

These figures are conceptual diagrams only; they do not introduce new numerical results. The generator script is provided as `04_supplementary_material/scripts/make_conceptual_vector_figures.py`.

## Citation-format note

Citation clusters have been distributed across the relevant claims and sentences. Numeric references were rebuilt from a clean auxiliary state so citation numbering follows first-citation order. Repeated citations may naturally reuse earlier numbers.


## Main-text density note

Extended diagnostic tables for baseline transparency, source acceptance, quota sensitivity, prediction-noise stress, and compact scalability checks are retained in the supplementary material rather than repeated in the main manuscript. The main text keeps the essential method, benchmark, configuration, and core result tables.

## Build note

Compile `02_latex_source/main.tex` with a standard LaTeX installation using the `elsarticle` class. The bibliography database is `references.bib`; `main.bbl` is also provided to preserve the validated bibliography used for the final PDF.

## Scope note

Bibliographic entries retain original source metadata, including publication venues and DOI strings of cited works. The manuscript metadata targets Applied Soft Computing.
