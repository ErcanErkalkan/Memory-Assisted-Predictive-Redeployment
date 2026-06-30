# LaTeX source

Compile from this folder. The source is prepared for Elsevier `elsarticle` and Applied Soft Computing.

Simple rebuild using the included `.bbl` file:

```bash
pdflatex ASC_PRR_NSGAII_manuscript.tex
pdflatex ASC_PRR_NSGAII_manuscript.tex
```

Optional full bibliography rebuild, if BibTeX is installed:

```bash
pdflatex ASC_PRR_NSGAII_manuscript.tex
bibtex ASC_PRR_NSGAII_manuscript
pdflatex ASC_PRR_NSGAII_manuscript.tex
pdflatex ASC_PRR_NSGAII_manuscript.tex
```

All figure and table files required by the main manuscript are kept in this folder so the paths remain simple and portable.
