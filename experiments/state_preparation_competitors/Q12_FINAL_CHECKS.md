# Final twelve-qubit numerical checks

These diagnostics retain the common target fixtures and IBM logical noise.
They neither launch the comparison matrix nor change a competitor's optimizer.
The initial protocol is frozen before collecting the new noisy results.

## Ballarin

Use the saved TCI startup checkpoints at depths 2, 3, and 4 from
`output/q12-tci-20260925/startup_depth_*.json`: twenty unchanged Adam updates
at the first interpolation coordinate, lambda=0.05. Score each resulting
logical circuit against the final Gaussian target constructed with published
TCI. These match the type of frozen checkpoint used in the interrupted earlier
compression probe; they are not fully trained lambda=1 circuits.

Evaluate each identical parameter vector with:

- Default: intersite/environment caps 64/16; tolerances `1e-9`/`1e-12`.
- Tighter: caps 128/32; tolerances `1e-11`/`1e-14`.

Use the existing autograd recomputation with interval 1 for both calculations
to limit memory. This changes stored intermediates, not contractions or updates.
Give each calculation its own 5,400-second deadline and 8 GiB physical/RSS
guard, recording terminated runs as unresolved rather than accuracy failures.

Report absolute fidelity difference, gradient norms, relative gradient
difference, cosine, discarded weights, ranks, runtime and memory. The local
numerical audit criteria are fidelity difference at most `1e-5` and relative
gradient difference at most 1%; these are our diagnostics, not paper settings.
A small difference between two compressed calculations is not proof of exact
noisy fidelity or a guarantee for later trained parameters. Additional trained
checkpoints, when computed, must be identified separately.

## RobustState

Freeze two depth-one checkpoints: twelve-qubit `haar_random_1` and
`tfim_critical`, root seed 300 with the existing depth-dependent seed split.
Use the final iterate after exactly 500 ideal Adam updates, learning rate
0.005, beta values 0.9/0.999 and epsilon `1e-8`. Preserve both moments and
the step counter when examining the first noisy update.

The ideal loss is still the Hilbert-Schmidt norm. For pure states,
`L = sqrt(2*(1-F))` and `gradient(L) = gradient(1-F)/L`; using this exact
identity avoids repeated dense ideal projectors. Tests compare its loss,
gradient and complete parameter path against the existing implementation.

At each fixed checkpoint, form the exact noisy density once and independently
draw 32 feedback estimates using the existing uniform-without-replacement
Pauli-basis sampling, 40 bases and 1,024 shots per basis. The raw shadow is
unchanged: no positivity projection or replacement by exact feedback.
Compare surrogate gradients and the actual next Adam updates, retaining the
warm moments. Report repeated-sample uncertainty, not only one cosine.

Score the exact-feedback diagnostic update and the first two prespecified
sampled updates using exact IBM logical density propagation. These one-step
scores are diagnostics, not a shortened noisy-training benchmark. All sampled
updates start from the same warm checkpoint; they are independent repeats,
not sequential optimization. No sample is selected by its observed fidelity.

No requirement is imposed that every stochastic gradient align with the exact
one. The finite-sample normalized Hilbert-Schmidt gradient is itself nonlinear
in the shadow, so its sample mean need not equal the exact-feedback gradient.
Report whether this budget has useful signal and stable updates here; a
failure to demonstrate adequacy is not evidence that the published algorithm
is intrinsically inferior. Two checkpoints do not establish convergence over
all targets, depths, and fifty noisy updates.

The [paper](https://arxiv.org/html/2311.16035v1) demonstrates the 40-basis
subsampling case at four qubits, and otherwise uses all Pauli bases with 1,024
shots each. At twelve qubits, the full-basis count is 531,441 and the shots per
update would be 544,195,584. The fixed 40-basis extrapolation must remain labeled.

## Execution

`q12_final_checks.py` runs individual probes and writes compact arrays plus
JSON receipts. `bounded_probe.py` supervises one process, preserving completed
results and recording time/memory termination explicitly. Run large workers
serially with numerical-library threads set to one. Do not overwrite prior
directories or combine completed and incomplete gradients into a comparison.

### Recorded resource-only amendment

The initial tighter depth-four job exceeded the per-tensor limit of `2**24`
elements after 189 seconds, before reaching the process memory guard. Retry
with an explicitly recorded larger local allowance, retaining the 8 GiB
process guard and unchanged compression settings. A different recomputation
interval may also be used if necessary; tests verify fidelity and gradient
parity. Keep each attempt in a separate directory. The initial diagnostic
source is archived with the run; optional resource arguments leave its
default numerical path unchanged.

The larger-allocation attempts at depths three and four subsequently reached
the 8 GiB process guard with interval one. Resource retries may use intervals
two and eight with the same local allowance and process guard. Tests cover
intervals one, two, four, and eight for ideal, IBM, and H2 execution. Retain
all failed attempts; never report their missing gradients as numerical zeros.

### Compact checkpoint storage

After the interval-only retries also exhausted memory, a separate ownership
audit found compressed factors retaining larger backing SVD arrays through
views. The optional `--compact-checkpoints` diagnostic copies outputs with
oversized backing storage into compact arrays; already compact tensors remain
shared. This preserves the mathematical
tensor, derivative, rank choices and truncations; a diagnostic trace reduction
can differ by floating-point roundoff. Tests compare complete fidelity and
gradients for ideal, IBM and H2 execution and verify the smaller retained
storage. The author SVD and production optimizer are unchanged. Preserve a
separate manifest and directory for compact-storage attempts.

### Streamed derivative storage

The compact-storage attempts also reached the memory guard at depths three
and four. A further execution amendment accumulates the SVD derivative terms
in one buffer in their original order, releasing full-matrix intermediates
after use. No SVD, regularizer, truncation or optimizer setting changes.
Eight frozen derivative fixtures from the prior kernel cover real/complex,
rectangular/square and degenerate/null inputs, alongside the independent
density and recomputation tests. Keep the prior kernel, new source hashes and
all attempts. Repeat default and tighter calculations under this revision;
compare completed defaults against the original gradients before accepting
paired compression results. The 8 GiB and 5,400-second guards remain fixed.

The resumed depth-three tighter run again reached the memory guard. A
subsequent temporary-disk preflight found only 187 MiB available and an actual
shell temporary-file allocation failed. The serial supervisor was interrupted
to terminate its active depth-four default worker and prevent the next large
retry. No disk offloading or unrelated file deletion was performed. Preserve
this host-resource stop separately from completed numerical comparisons.

### Remote resource and measurement-budget amendment, 2026-09-27

Use the existing `gate` SSH alias (the gateway reports hostname
`georg.wias-berlin.de`) and a separate source snapshot on `leonhard-13`.
The capacity check found approximately 681 GiB available RAM and 144 logical
CPUs. This is an observed availability snapshot, not a scheduler reservation.
No scheduler executable was found in the remote shell path.

Repeat depth-three and depth-four default/tighter pairs on this one host,
using the original frozen checkpoints and TCI construction. Keep workers
serial, numerical threads at one, recomputation interval one, compact
checkpoint storage enabled, and the local allowance at `2**26` elements.
The remote process RSS guard is 128 GiB and the deadline remains 5,400 seconds
per worker. The increased memory allowance is an execution-resource change;
no rank, tolerance, objective, noise or optimizer change is permitted. Preserve
new source hashes and environment receipts. Compare each completed pair on
the same platform, and report cross-platform differences separately.

Reuse the earlier 40-basis RobustState audit. The next declared budget is
400 uniformly sampled Pauli bases, still without replacement, with 1,024
shots per basis and 32 independent repeats at each of the same two trained
checkpoints. Rebuild the unchanged 500-step warm stage and compare its saved
parameters, moments and target against the old checkpoint; do not attribute
a changed warm state to the measurement budget. Different budgets receive
explicit seed streams; the default 40-basis stream is unchanged.

Run these two target probes independently, each with a 16 GiB RSS guard and
5,400-second deadline. The larger sampling budget is a stated benchmark
extension of the published procedure, not a published twelve-qubit setting.
Report gradient/update comparisons, confidence intervals and the predefined
one-step fidelity diagnostics. A positive mean gradient cosine with an interval
above zero provides evidence of directional signal at that checkpoint, not a
guarantee of convergence or adequacy across all targets. Do not automatically
increase the budget, change sampling, or launch the full comparison campaign.

The first remote 400-basis feedback calculation took 318 seconds, predicting
more than 90 minutes for 32 serial repeats. A resource-only amendment splits
the same 32 globally indexed trials into four disjoint ranges of eight per
target. `--trial-start` preserves the exact seed assigned to each trial. Tests
compare a separate range against the matching rows of a serial calculation.
Retain global trials zero through seven from the original worker, stopping it
after those complete, and run ranges 8-15, 16-23 and 24-31 independently in a
separate snapshot. Do not duplicate trials in the combined estimates; require
all 32, matching warm parameters/moments and exact-feedback references before
reporting a completed budget audit. Each batch retains the 16 GiB and
5,400-second guards. No sequential training update is split or reordered.

### Depth-four local allocation allowance, 2026-09-27

The remote tighter depth-four attempt stopped after 1,766.83 seconds with
`Comb local tensor exceeds max_tensor_elements`. It did not exhaust the
128 GiB process budget: sampled peak RSS was 6.12 GiB. Preserve this failed
attempt and its traceback without assigning a fidelity or gradient result.

Retry this one calculation in a separate output directory with the local
allocation guard increased from `2**26` to `2**28` elements. A complex128
tensor at the new limit is 4 GiB; the independent 128 GiB process and
5,400-second wall-time guards remain. Keep all compression ranks, tolerances,
TCI targets, source files, frozen parameters, IBM logical noise, checkpoint
storage and one-thread execution unchanged. `_allocation` only checks sizes
and records diagnostics, so this raises a resource ceiling without changing
the tensor algorithm. No dense fallback is allowed. Reuse the completed
remote default, whose largest local tensor has 13,298,688 elements, below
both ceilings. Its comparison partner must still match target, parameter and
source hashes. The fidelity and gradient acceptance limits are unchanged.
