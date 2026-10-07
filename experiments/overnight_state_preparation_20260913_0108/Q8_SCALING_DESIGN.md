# Conditional q8 scaling study — design only

This document authorizes no target loading, construction, simulation or confirmation inspection. Implementation and execution require a new protocol, tests, source freeze and allocation after the coordinator's confirmation decision. The completed q6 physical follow-up validates a weighted trajectory backend at identical inputs. The strongest composed state-preparation candidate instead uses exact density objectives, exact finite-deletion ranking and exact repair. Substituting MPS trajectories changes its numerical objective and branch selection. A backend diagnostic cannot establish scalability of that full algorithm.

## Track A: a small backend accuracy and resource diagnostic

Use two canonical q8 fixture identities, Gaussian mu0.5/sigma0.1 and Haar random1, only if those exact entries exist in the immutable target catalog. Missing entries abort this proposed panel; do not generate replacement targets. Their status is engineering development, not additional confirmation evidence. Freeze their vector hashes only after allocation. Until then, load no vectors. The two targets exercise structured and demanding entanglement cases without choosing cases after seeing backend behavior.

Build each complete zero-input conditional initializer using the confirmed, unchanged initializer rule, lowering and compiler. Charge its physical construction once and show the same declared construction charge for every compared backend. Preserve the complete physical circuit, angle maps and native gate identities. This input matches the composed candidate's initialization; the following three updates do not reproduce its structural algorithm.

Use the same two fixed training channels, depolarizing_1s_all and dephasing_1s_all, with an equal-mean weighted estimator. Predeclare exactly **three** fresh Adam updates at rate .02, beta1 .9, beta2 .999, epsilon 1e-8 and four weighted paths per channel per update. Grant identical paths, weights and persistent fresh-start moments to:

1. Dense q8 statevector weighted adjoint oracle.
2. MPS weighted adjoint, exact-zero truncation threshold and no hard cap.
3. MPS weighted adjoint, threshold zero and one declared working-state cap **8**.

The exact target MPS remains untruncated in all arms. Truncating the target would change the benchmark. In the capped arm the exact target and initial adjoint may therefore have bonds above the working cap before subsequent operators trigger compression; record target, forward and adjoint dimensions separately. The public truncated adjoint does not differentiate the SVD truncation decisions. Its difference from the dense gradient is an approximation-bias measurement, not a claim to be the exact derivative of the truncated numerical loss.

Do not search caps, thresholds, update counts or learning rates. Cap8 is deliberately below the q8 exact maximum internal bond16. The exact arm must finish first; a failure blocks interpretation of the capped arm. No removal of tiny gradients, phase-coordinate optimization change, extra restart or best-checkpoint selection is allowed.

### Paired numerical checks

First replay one already authenticated q6 step unit per development target with the unchanged maps/weights and compare to the completed physical follow-up. This is an integration overlap, physically charged again and not counted as new replication. Require bit-preserved source input and identical map digests; require original same-theta gradient, fidelity, phase-aligned state and norm errors <=1e-10.

At q8, compare all gradient coordinates and path fidelities at each identical dense pre-update vector. Also retain each backend's actual independent optimizer trajectory. After every saved update, compare dense and MPS physical outputs at identical theta, then compare outputs at their independently updated theta. Exact-arm same-theta tolerances remain 1e-10. Keep raw theta differences; evaluate the same analytically derived Pauli-rotation telescoping bound plus an explicitly frozen 2e-10 propagation allowance for different theta. Remove only provable first-RZ zero-input global phases, with prior gate and noise supports included in the proof. Record the unaligned vector distance as well.

The no-cap arm must discard exactly zero weight, have internal bonds <=16 and satisfy the same norm gate. For the cap8 arm, record every split and accumulated discarded squared weight, norm drift, same-theta gradient maximum/norm/cosine discrepancy, physical state distance and target-fidelity error. Do not call the sum of discarded weights a certified global fidelity bound. Keep absolute gradient differences when the reference gradient is nearly zero; no small-gradient exclusion is permitted.

### Fresh endpoint paths and paired uncertainty

Use a new namespace for final diagnostics, independent of all training, q6 overlap, development debugging and checkpoint work. Freeze **32 independent final batches**, each with four weighted paths per channel. The same batch operators and weights are paired across all three final iterates, and each batch is evaluated by the dense statevector oracle. This separates optimizer differences from MPS evaluation bias. Retain the two channel estimates individually and their equal mean.

Compute each paired difference at the batch level, preserving its analytic no-factor weight and conditional noisy-path weights. Report the mean, sample standard deviation, standard error and an explicitly labeled approximate two-sided 95% Student interval with 31 degrees of freedom. Paths inside a weighted batch are not independent observations. Publish all 32 paired differences and noise-stream identities. The fixed diagnostic sample count may be inconclusive; it never expands in response to a promising mean. Intervals are descriptive approximation checks, not a benchmark-win claim.

These endpoint scores concern only the two training-channel expectations. Any later claim about overall state-preparation quality requires the full unchanged all12 benchmark evaluation and its noise-family aggregation; this backend diagnostic alone makes no such claim. A capped approximation remains merely investigational unless its paired final fidelity loss is demonstrably small and its measured time or memory benefit is material. A possible later promotion rule must be frozen before execution, for example an upper confidence bound on equal-mean fidelity loss <=0.1 percentage points plus at least a factor-two reduction in training CPU or actual retained tape storage. Do not tune this rule or infer a speedup from the qubit count alone.

### Honest memory and clocks

Run at most one numerical worker at a time, with every BLAS/OpenMP setting fixed at one. Use fresh worker processes for the backend resource comparisons and retain process peak RSS in the platform's actual units. A process high-water mark is not incremental backend memory. Include target loading/factorization, sampler construction, map generation, gradients, all forward/adjoint work, updates, diagnostics, failed attempts and checkpoint I/O in an inclusive attempt ledger, while retaining nonadditive component timers.

Measure the complete retained forward-tape tensor storage where available, deduplicating shared array buffers, and report the exact scope. Report simultaneous adjoint/tangent storage only if it is actually observed. Current maximum-individual-MPS-tensor bytes are insufficient evidence of total tape memory. Record target bonds, forward/adjoint peaks, every truncation decision, dense vector bytes and actual RSS. Dense conversion for physical checks is separately timed and must not contaminate an asserted MPS-only training-memory measurement. The dense fixture input and its factorization remain part of honest end-to-end resources; this study does not establish polynomial scaling for arbitrary dense target data.

Persist an attempt marker before each batch, complete raw maps/weights/source hashes, full optimizer moments, theta and RNG identities. Reuse only authenticated complete units. Catchable interruption work is retained and charged; an unobserved hard kill blocks automatic resume until its unknown cost is addressed. A fixed 120 CPU-second per-case safety budget can stop at a durable unit boundary; reaching it yields an incomplete diagnostic, not an accepted abbreviated experiment. Never infer unmeasured work as zero.

## Track B: preserve the complete candidate at q8

A direct test of the same candidate can retain exact density propagation, exact two-channel deletion ranking, the fixed shortlist, symmetric repair, optimizer, compiler, initialization and all12 final evaluation. At q8 a complex128 density matrix contains 65,536 elements, approximately 1 MiB; this elementary storage count excludes tapes and temporaries. It is insufficient to predict the complete circuit's memory or runtime.

After confirmation, first measure one bounded exact objective/gradient and one exact finite-deletion ranking on a frozen q8 start, together with the corresponding strongest matched control operation. This checks whether the original numerical mechanism is feasible. Such a probe is a separately charged diagnostic, not a shortened performance comparison. If a subsequent full run is affordable, preregister one full q8 candidate/control panel using the complete existing 3000 individual-channel call cap, 150 joint-objective warmup calls, 25 joint-objective repair calls, shortlist2, maximum15 deletion rounds, and all failed-branch costs. Use the actual strongest objective-matched control selected by the completed confirmation protocol, not a weaker convenient optimizer. Retain final iterates, original initialization/full-procedure charges and common-total-CPU availability; report unavailable cutoffs rather than replacing them with later checkpoints.

Reducing the number of rounds or changing the objective, repair budget or optimizer for convenience creates a new candidate and must be labeled accordingly. Replacing exact deletion/repair with finite trajectory estimates also creates a new candidate: it needs common-map finite-deletion tests, explicit estimator/selection-bias treatment, independent endpoint evaluation and matched stochastic baselines before any scaling claim.

Track A can establish larger backend agreement and expose approximation costs. Track B can test whether the confirmed complete candidate remains feasible at q8. Neither two-target experiment establishes an asymptotic scaling law or broad state-of-the-art performance.
