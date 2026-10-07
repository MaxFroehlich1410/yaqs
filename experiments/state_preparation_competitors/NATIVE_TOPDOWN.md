# Native-noise top-down pilot

Run `python -m experiments.state_preparation_competitors.run_native_topdown --output <new-directory>`
in the pinned noisy-competitor environment with one numerical thread.

The pilot uses Gaussian, TFIM critical and Haar 1, six qubits, optimizer seed 300,
and the existing synthetic IBM coefficients. It reuses authenticated noiseless
online-Krotov/L-BFGS construction and pruning B inputs. Native refinement starts
from the pruned parameters, before the previous logical-noise refinement.
Noiseless construction and pruning are extended along the same lineage if the
new depth search needs checkpoints beyond the archive.

Compile each symbolic circuit once with the common RZ/SX/X/CX compiler. Preserve
original parameter indexing, affine scales and offsets; require exactly one
native occurrence per original trainable parameter. Compilation can reorder
chronological online updates. The update formula and budget are unchanged, but
this is an adaptation to the native representation, not an identical online
parameter path. Numeric endpoint compilation previously eliminated additional
fixed gates; those earlier schedules are different and remain historical context.

Noise follows every native operation, including fixed and zero-angle RZ gates.
Each gate window has at most one Pauli jump, with probabilities
`p0 = exp(-sum(rates))` and `pj = (1-p0)*rate_j/sum(rates)`. Rates per acted site
are X/Y/Z each `1e-4`; two-qubit windows additionally have XX/ZZ each `0.0015`.
There is no idle or measurement noise. This is a synthetic profile rather than
a calibrated physical device; in particular, RZ gates are charged noise here.

Noisy refinement reuses the historical optimizer: 200 updates, the first 100
online, then 100 batch cross-trajectory updates, fixed rate 0.2, final-iterate
retention. L-BFGS remains confined to noiseless construction. The historical
cross signal is not asserted to be the gradient of mean sample infidelity.
M=100 unconditional trajectories use the same seed convention (base zero,
counter 5300), fixed common random numbers, and exact empirical compression of
no-jump duplicates. A missing-jump bank fails that candidate without resampling.

Each depth receives a separate native validation bank of 32,768 unconditional
trajectories. Its entropy ends with a distinct native-validation role 91000.
Use the original material-improvement threshold
`0.001 + 2*hypot(SE_current, SE_record)`, patience three, depth cap sixteen.
The winner is the maximum validation fidelity among visited eligible candidates,
with earliest depth breaking exact ties. The old logical winner and stopping
depth are not reused. Lock all decisions before exact Qiskit density-matrix
assessment. Exact assessment integrates all modeled jump histories, so it has
no Monte Carlo test error and cannot overfit the training bank.

Preserve the trained native schedule through validation, storage and assessment;
never recompile bound parameters. Save raw validation codes in compressed binary,
reference authenticated construction histories in place, and keep fine-tuning
curves, final parameters, seeds and noise events in sealed per-depth bundles.
Charge all visited construction/pruning CPU, including reused historical work,
plus fresh compilation/refinement and validation. Report newly spent wall time
separately. Existing native-trained NACL, RobustState and supported Fontana
results can be reused with their original locked selections. Historical fixed
and layerwise controls were logically trained and are outside this matched
native-training comparison.
