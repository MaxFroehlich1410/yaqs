# Portable manuscript data

The numerical files described here stay local and are excluded from the
GitHub source checkout. Supply the local snapshot to regenerate the manuscript.

`results.json` contains the exact selected endpoint values used by the paper,
the 120 assessed logical candidate records needed to check selection, target
metadata, protocol constants, exclusions, and source SHA-256 fingerprints.
`logical_endpoints.csv` and `native_endpoints.csv` contain the same endpoint
values at full stored precision. Fidelity is stored as a fraction, not percent;
CPU is measured in seconds. Figures and tables format these values for display.

The logical source is the completed nine-target six-qubit experiment in
`output/logical-noise-all-q6-ibm-20260920/`. The native source is the three-target
pilot in `output/native-topdown-ibm-pilot-20260920/`. All source paths in the JSON
are relative to the YAQS repository root. Source fingerprints identify the
artifacts used to export this snapshot; no paths are opened during a portable
figure build.

The 120 logical candidate records comprise 9 frozen Top-down endpoints,
27 NACL checkpoints, 18 RobustState circuits, 12 Fontana circuits, 18 fixed VQSP
circuits and 36 layerwise circuits. This is not a count of all historical
Top-down depth candidates or discarded structural proposals. Endpoint CPU
includes the complete historical/current construction and training search for
that method and target. Sum it once per method--target group. The CSV contains
one row per group; the candidate array deliberately omits CPU to avoid charging
the same search repeatedly.

The Top-down depth winners remain fixed from their source archives. The later
8,192-trajectory logical validation is diagnostic for those endpoints. The
critical TFIM RobustState endpoint also retains its original validation winner,
even though another assessed depth has a slightly larger exact fidelity.
Missing Fontana/Haar entries mean unsupported targets and are never zero scores.

The snapshot supports reproducing all manuscript figures and tables without
YAQS, Qiskit, or the training environment:

```sh
python tools/build_figures.py
python -m unittest discover -s tools -t . -p 'test_*.py'
```

Run from the manuscript directory with NumPy and Matplotlib installed. This
snapshot is not a standalone training reproduction package. Full parameter
histories and recursively referenced original archives remain in the repository
and must be preserved separately. Exact endpoint values integrate the stated
channel; they do not quantify optimizer-seed variability or unseen-target
generalization. The real-target and Haar summaries are descriptive groups,
not a controlled test of state realness.

`q12_iqsp_resource_diagnostic.json` separately records the preliminary twelve-qubit
Gaussian resource observation in the Discussion, with source hashes, compression
settings and hardware context. It measures one gradient at an early checkpoint,
not completed training or a matched comparison. It does not contribute to
`results.json`, any figure, or any competitive ranking. The quoted memory is
sampled peak physical footprint, including compressed memory, rather than RSS.
