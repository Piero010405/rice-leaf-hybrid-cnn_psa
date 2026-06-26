# Rice Leaf Disease IEEE Paper Project

This folder contains the initial IEEE-style LaTeX structure for the paper:

**Experimental and Descriptive-Comparative Evaluation of Segmentation and Explicit Texture Features in Lightweight CNNs for Rice Leaf Disease Classification**

Author: D'Alessandro del Piero Sarmiento Ayala  
Email: d.sarmiento@usil.pe  
ORCID: https://orcid.org/0009-0005-1480-1671  
University: Universidad San Ignacio de Loyola  
Course: Matemática Aplicada en la Computación  
Instructor: Emanuel Yarleque Medina

## Structure

```text
rice_leaf_ieee_paper/
  main.tex
  sections/
    00_abstract.tex
    01_introduction.tex
    02_related_work.tex
    03_methodology.tex
    04_experimental_development.tex
    05_results.tex
    06_discussion.tex
    07_conclusions.tex
    08_reproducibility.tex
  bib/
    references.bib
  figures/
  tables/
  results/
  code_appendix/
  build/
```

## How to compile

Using `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

Using manual BibTeX:

```bash
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

If the manual BibTeX command cannot find `main.aux`, compile from project root and check your LaTeX installation paths.

## Notes

- The paper is intentionally skeletal. Sections contain TODO markers to guide the next writing phase.
- Some references from the preliminary review still require exact metadata verification.
- Final experimental tables should be copied from `outputs/tables/multiseed_summary_mean_std.csv`, `paired_deltas_vs_E0.csv`, and `wilcoxon_vs_E0.csv`.
