# Top-down quantum state preparation under gate noise

First manuscript draft based on the completed six-qubit logical-noise comparison and the separate native-gate pilot. The main text follows the approved six-section structure and the supplied `PAPER_WRITING.md` guide.

## Source-only GitHub checkout

The GitHub branch contains manuscript source and generation tools. Numerical
result snapshots in `data/`, generated files in `figures/` and `tables/`, and
compiled PDFs stay local. Compilation and regeneration require those local
artifacts; a fresh clone does not contain a ready-to-compile paper package.
The manuscript data tests skip when the local `data/results.json` is absent.
The package instructions below apply after supplying the local artifacts.

## Compile in Overleaf

1. Upload the supplied source ZIP as a new Overleaf project, or upload this directory while preserving its subdirectories.
2. Set **Main document** to `main.tex` and **Compiler** to **pdfLaTeX**.
3. Use a current TeX Live version. REVTeX 4.2 and the required packages are available in standard Overleaf installations.
4. Recompile. Overleaf runs BibTeX automatically; the bibliography file is `references.bib`.

No external images, shell escape, repository imports, training archives, or network requests are needed to compile the paper. All three figures are included as vector PDFs. The `tables/` files are ordinary LaTeX inputs.

For local compilation with TeX Live and `latexmk`:

```sh
sh compile.sh
```

The output is `main.pdf`. A compiled preview and clean source ZIP are also delivered separately.

## Contents

- `main.tex`: document preamble, abstract, and section order.
- `authors.tex`: author block, isolated for easy editing.
- `sections/`: six main sections and mathematical/numerical appendices.
- `references.bib`: verified BibTeX entries.
- `CITATION_NOTES.md`: primary-source checks and the unresolved public identifier for the supplied Krotov manuscript.
- `figures/`: workflow, fidelity, and fidelity-versus-CPU figures, with PDF publication files and PNG previews.
- `tables/`: two main tables and two supplemental tables.
- `data/`: portable numerical inputs and provenance for the figures and tables.
- `tools/`: figure/table generation and numerical regression checks.

## Regenerate the visual material

Compilation uses the supplied PDFs and table files. To regenerate them, install Python, NumPy, and Matplotlib and run:

```sh
python tools/build_figures.py
python -m unittest discover -s tools -t . -p 'test_*.py'
```

The generator reads the portable data snapshot. It does not launch training, reselect circuits, or require the experiment archives. The snapshot retains the exact endpoint values; rounding is applied only for presentation.

## Items for the authors to resolve

- The author list is Maximilian Fröhlich, Tomasz Szoldra, Martin Eigel, Michael Hintermüller, and Peter Schmelcher, in the requested order. WIAS affiliations follow the supplied recent manuscripts. The Hamburg affiliations follow the official profiles of [Tomasz Szoldra](https://www.physik.uni-hamburg.de/en/iqp/schmelcher/personen/szoldra.html) and [Peter Schmelcher](https://www.physik.uni-hamburg.de/en/iqp/schmelcher/personen/schmelcher.html). Confirm any additional affiliations specific to this paper before submission.
- Add applicable funding acknowledgments. No grants or author approvals have been inferred.
- Replace the `Froehlich2026HybridKrotov` author-supplied-manuscript entry with its public identifier when available.
- Assign a public deposit or release for the full experiment archive before making a corresponding data-availability claim. The draft does not invent a DOI or imply that all local histories are already public.
- Confirm the experiment host's CPU model if a machine-specific runtime statement is desired. The recorded numerical versions and worker/thread policy are reported; the source manifests do not identify an exact CPU model.

These are editorial and release metadata items. The paper's scientific text, figures, tables, and bibliography compile without unresolved LaTeX placeholders.

## Scope preserved in the draft

The evaluated method uses every-depth construction, pruning B, and 100-history noisy refinement. L-BFGS and best-iterate restoration apply to noiseless construction; noisy refinement retains its final iterate. The cross-trajectory signal is not assumed to be the sampled-loss gradient. The single-jump synthetic Pauli channel, original elementary-gate noise windows, independently selected endpoints, unequal finite budgets, development-used targets, and one optimizer seed remain explicit.

The native-gate pilot remains a main-text result. Historical pruning D, M=400, first-90% stopping, and other algorithm variants are not pooled with the current results. The manuscript makes no hardware-speedup, scaling, or general-superiority claim.

## Archive preservation

This manuscript directory is a separate export package. The earlier `manuscript/main.tex` and every experiment output archive are preserved.

The portable data support the paper's figures and tables, not full optimization replay. Retain these repository archives and all histories that they recursively reference:

- `output/logical-noise-all-q6-ibm-20260920/`
- `output/logical-noise-three-target-ibm-pilot-20260920/`
- `output/topdown-batch-refinement-q6-20260919/`
- `output/native-topdown-ibm-pilot-20260920/`

The complete source index is `experiments/state_preparation_competitors/PAPER_HANDOFF.md`. The executed all-target protocol is `experiments/state_preparation_competitors/LOGICAL_SUITE.md`, together with its manifest. The runner alone is not a standalone optimization-reproduction package.
