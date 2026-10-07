# q12 backend diagnostic: deferred

No q12 target was loaded, constructed, propagated or scored for this planning
document. No q12 code or scientific protocol was prepared. The selected complete
method failed its fixed q6 confirmation primary gate. Larger-system science is
therefore stopped; full-budget results cannot replace that gate. An exceptionally
clear resource advantage would require its own complete evidence and coordinator
decision, and is not assumed here.

The fixed fixture supports q6 and q12. A q8 scientific intermediate would require
new targets and is ineligible under the immutable benchmark. The following route
is a prospective backend diagnostic only. It is not an implementation or a
confirmation of the complete exact-density structural L-BFGS method.

## Current implementation boundary

The [rank-two synthesis guard](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/synthesis.py:39)
explicitly accepts at most 64 amplitudes. The frozen lazy initializer and parent
construction wrappers likewise enforce their existing scientific domains. They
must not be patched or bypassed to execute q12. A separately named adapter would
need to reproduce the q1-q6 primitive exactly before extending its validated
domain. Merely changing a size guard would not establish numerical equivalence.

The [statevector trajectory backend](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/src/mqt/yaqs/optimization/statevector.py:103)
has no q6-only size guard and applies local matrices without embedding a full
unitary. The [MPS replay bridge](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/mps_bridge.py:168)
uses the public single-trajectory adjoint and identical unitary noise maps and
weights. It physically starts from zero, can receive a pre-factorized MPS target,
and reports actual bonds, discarded SVD tails and operator work. The underlying
MPS backend alone is not evidence that it outperforms the dense implementation.

## Smallest prospective diagnostic

If separately justified and allocated, preregister the same four development
target identities at q12, all retained regardless of target overlap. Use seed11
and a new `q12_backend_overlap` namespace. The first experiment should study a
single deterministic rank-two physical initializer per target, with its TT
overlap reported even when below 0.99. Such a probe is explicitly not the complete
conditional initializer: it does not execute or pretend to replace the legacy
570-update fallback.

A new tested adapter could reproduce sequential TT-SVD and exact local Weyl
lowering with the original conventions. At q12 its raw preparation would contain
135 one-qubit and 33 two-qubit scalar gates, all including physical initialization
and zero-angle factors. Independently check ideal statevector agreement, and then
apply the unchanged generic endpoint compiler with its cost and parameter map
recorded. Any compiler or source-copy work is charged. Do not introduce arbitrary
unitaries as free noisy operations.

Replay four fixed weighted paths per training channel, eight total, through both
dense and untruncated MPS backends. Compare every path fidelity, weighted loss and
gradient, then three fresh-stream Adam updates at rate0.02 with complete moments
and counters. The finite path mixture is shared exactly; it is not an exact
channel expectation or a final optimization comparison. Require all absolute
loss/gradient/parameter discrepancies at most1e-10, exact zero discarded tail,
all cases complete and maximum physical internal bond at most64. A mismatch is
a backend failure to investigate on engineering fixtures, not a reason to change
the target panel or loosen tolerances after observing results.

Factor each dense target into an untruncated MPS once per case and charge it
separately. Record fixture input, TT construction, lowering, compilation, target
factorization, sampling, dense replay, MPS replay and checkpoint costs. Persist
fixed maps and weights, full current Adam state, compact audit summaries, and
the raw audit for the current operation. Unknown killed work has null cost and
prevents an ordinary successful resume.

## Bounds and interpretation

A q12 complex128 statevector is64KiB. Its exact density matrix alone is256MiB;
a tape of169 such matrices would already occupy42.25GiB, before adjoints and
temporary arrays. Thus a q12 trajectory diagnostic cannot be presented as direct
scaling of the selected dense exact-channel optimizer and deletion oracle.

The Hilbert-space MPS bond bound is64. At that bound, one fully populated q12
MPS has10,920 complex entries, or174,720 bytes, exceeding one dense statevector.
These counts omit tapes, SVD workspace and Python overhead. Low observed bonds
may help; the existence of an MPS representation does not guarantee a memory or
runtime advantage. An eight-path dense forward tape for169 vector positions is
about84.5MiB before additional work. Serialize MPS paths and use one worker with
all four native thread limits set to one.

A future protocol should admit at most four cases, cap each at600 seconds wall
and2GiB process RSS, reserve at least512MiB free output space, and stop admission
before the reporting reserve. These are proposed bounds, not an allocated run.
Measure actual process peak RSS as well as individual MPS tensor storage. A
supervisor must preserve interruption/kill markers and observed wall duration;
it must not invent missing CPU. Any cap breach remains an explicit failed case.

Only after exact overlap passes would a single separately declared bond16 cap
probe be meaningful. It must replay the same maps and parameters, compare state,
loss and update deviations with both untruncated backends, and report all actual
discarded tails. The existing truncated adjoint does not differentiate SVD
truncation decisions, so its signal must not be labelled the exact gradient of
the truncated computation. No capped optimization or twelve-profile performance
claim follows from that probe. This entire route remains deferred.
