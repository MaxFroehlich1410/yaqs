# Hybrid Krotov compression-screen findings

Completed 2026-09-06. Protocol: `yaqs.compression_hybrid_screen.v1`.

The subsequent [25/30-update extension](state_preparation_hybrid_extension_findings.md)
finds a stronger mean at 25 online updates and rate 0.7. The results below
remain the unchanged record of this earlier screen.

## Main conclusion

The strongest tested construction pipeline uses **20 online updates at rate
0.5 at the start of each layer-growth stage, followed by batch training**.
Its mean exact noisy fidelity is **74.9261%**, compared with **71.5027%** for
the previous batch-only pipeline: **+3.4234 percentage points**. It improves
all four target-family means and all eight target/seed pairs.

This supports investigating an aggressive early hybrid schedule. It does not
yet attribute the whole gain to online ordering: the largest improvement also
uses a larger early learning rate than the batch control. The same-rate hybrid
comparisons have much smaller, mixed gains. Noisy warm refinement remains
slightly worse on average than matched noiseless refinement at every setting.

## Scope, execution, and verification

The [predeclared screen](state_preparation_hybrid_pilot.md) used four fixed
six-qubit states, two seeds (11, 29), the same restored target fixture and
initial parameters as the [batch pilot](state_preparation_pilot.md), and the
same calibrated exact local Pauli channel. No checkpoint was selected by its
evaluation fidelity. All nine hybrid settings and the batch control completed.

- 80 jobs, 320 exact-channel endpoints, 1040 training blocks.
- 53600 optimizer updates, of which 5040 were online. This counts work in both
  the early-construction and separate warm-refinement comparisons.
- Eight single-threaded process workers; wall time **1144.9 seconds / 19.1 minutes**.
- All 1122 JSON artifact checksums verified. Recomputed aggregate results match
  the saved summary. Source fingerprints, phase boundaries, per-step learning
  rates, total update budgets, fixed final gate counts, and identical warm
  starts were verified after completion.
- The recomputed batch controls reproduce the original parameter vectors to
  a maximum absolute difference of **1.64e-15**. Their circuits and exact noisy
  fidelities agree within the required 1e-11 tolerance.
- All **316 optimization tests passed**, including uninterrupted-versus-chunked
  hybrid training at switches 5/10/20, high-noise fixtures, interruption on both
  sides of the switch, and multi-target serial/parallel equivalence.
- New experiment files pass their applicable lint/type/format hooks. Full
  repository lint still reports the pre-existing manuscript/plotting/MPS-test
  issues, broken BibTeX Node dependency, and missing matplotlib in the lint
  environment. Unrelated hook-generated edits were reverted.

Artifact directory:
`/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-hybrid-compression-screen-20260906`.
The original batch source is retained separately at
`/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-compression-pilot-20260905`.
The manifests contain the full configuration and source checksums; each
setting/target/seed directory contains `result.json` and `blocks/*.json`.

## Complete setting comparison

All entries below are **percent noisy fidelity**, computed by exact density
matrix evaluation after pure-state MPS training. Higher is better. Target
families have equal weight, with two seeds averaged within each family.

`Cold final` means hybrid during layer growth, followed by unchanged magnitude
pruning, batch-only repair, and batch-only final continuation: 570 total
updates. `Warm 0` and `Warm N` are separate 50-update noiseless/noisy refinements
from the **same original batch-trained M checkpoint**, not from the hybrid
construction. Do not combine their gains or describe warm N as noisy refinement
of the new best hybrid construction; that combined pipeline was not tested.

| Online prefix | Online rate | Cold final    | Gain over B, pp | Positive cold pairs | Warm 0        | Warm N        |
| ------------- | ----------- | ------------- | --------------- | ------------------- | ------------- | ------------- |
| B: none       | —           | 71.502713     | reference       | —                   | 71.502713     | 71.490518     |
| 5             | 0.1         | 71.482664     | -0.020049       | 6/8                 | 71.502955     | 71.490760     |
| 5             | 0.2         | 71.559652     | +0.056939       | 5/8                 | 71.528648     | 71.518901     |
| 5             | 0.5         | 71.688603     | +0.185891       | 7/8                 | 71.606609     | 71.599109     |
| 10            | 0.1         | 71.531301     | +0.028588       | 5/8                 | 71.503204     | 71.490965     |
| 10            | 0.2         | 71.542380     | +0.039667       | 4/8                 | 71.554271     | 71.541804     |
| 10            | 0.5         | 72.712162     | +1.209449       | 6/8                 | 71.706184     | 71.685840     |
| 20            | 0.1         | 71.295671     | -0.207042       | 1/8                 | 71.503718     | 71.491398     |
| 20            | 0.2         | 71.555960     | +0.053247       | 5/8                 | 71.604660     | 71.588589     |
| 20            | 0.5         | **74.926143** | **+3.423431**   | **8/8**             | **71.896139** | **71.856826** |

The complete raw aggregates also retain the pre-final-continuation
`cold_construct` endpoint at update 520. No unfavorable setting was omitted.

## Target-wise results for the strongest cold setting

Compare H20 at online rate 0.5 with the batch control after the full 570-update
pipeline. Both have 78 one-qubit and 30 two-qubit gates.

| Target            | Batch fidelity, % | Hybrid fidelity, % | Gain, percentage points |
| ----------------- | ----------------- | ------------------ | ----------------------- |
| Gaussian          | 80.709135         | 81.401798          | +0.692664               |
| Critical TFIM     | 80.216672         | 80.660929          | +0.444257               |
| Haar-random       | 44.003510         | 55.731317          | +11.727807              |
| Random bond-2 MPS | 81.081533         | 81.910528          | +0.828995               |

Every paired gain is positive; the smallest is +0.1412 pp and the largest
is +17.2710 pp. The aggregate improvement is nevertheless dominated by the
Haar target: its contribution is about 86% of the summed target gains.
Do not portray +3.42 pp as a uniform gain on every family. The other three
family means improve by approximately 0.44–0.83 pp.

## Evidence for faster early optimization

The following are mean **noiseless training fidelities in percent during the
first depth-one growth stage**, before pruning. They are descriptive checkpoints,
not checkpoints used for selection.

| Depth-one update | Batch, rate 0.2 | H20, online rate 0.5 then batch 0.2 |
| ---------------- | --------------- | ----------------------------------- |
| 1                | 4.426800        | 4.923323                            |
| 5                | 5.888978        | 9.491077                            |
| 10               | 8.212082        | 15.157648                           |
| 20               | 13.148664       | 20.754230                           |
| 100              | 46.723082       | 55.172458                           |

The faster early progress is visible directly, and an advantage remains after
the subsequent 80 batch updates. Thus the improved final result is accompanied
by better early fitting, rather than appearing only after noisy evaluation.
The same-rate H20 at 0.2 reaches 13.214191% at update 20 and 46.621971% at
update 100, close to the batch control. This reinforces the need to separate
larger early steps from gate-sequential ordering in the next causal test.

## Warm refinement and noise awareness

At the strongest warm schedule (20 online updates at 0.5, then 30 batch
updates at 0.1), noiseless refinement reaches 71.896139% and noisy refinement
71.856826%. Both improve on their respective batch counterparts, but noisy
refinement loses to matched noiseless refinement by **0.039313 pp**.
The corresponding noisy-minus-noiseless mean is negative for all ten settings.

This does not demonstrate that noisy optimization cannot help. It shows that
the tested R=8, 50-update independent-trajectory refinements have not added
an average benefit beyond matched noiseless training on these saved starts.
It also explains why a short hybrid test performed only on a mature warm
checkpoint could miss the more useful early-construction effect seen here.

Measured mean training-block times for B versus H20/rate-0.5 were about
57.0 versus 54.9 seconds for construction plus final continuation, 3.87 versus
4.13 seconds for warm noiseless refinement, and 43.51 versus 43.26 seconds
for warm noisy refinement. These are concurrent-worker timings, not evidence
that hybrid sweeps intrinsically cost less. Scheduling, core type, and other
machine activity can affect those measurements. Equal update budgets do not
imply equal computational work; public-API metric evaluations are included.

## Revised leading pipeline and remaining questions

The strongest tested complete pipeline is now:

1. Grow BMPD through depths 1, 2, 3, 4. At each depth, use **20 online
   noiseless updates at 0.5, then 80 batch noiseless updates at 0.2**.
2. Keep the original six magnitude-pruning rounds: remove ten one-qubit and
   five two-qubit gates per round, with **20 batch noiseless repair updates
   at 0.1** after each deletion batch.
3. Finish with **50 batch noiseless updates at 0.1** and evaluate the fixed
   endpoint under the calibrated noise channel.

No noisy training is part of this best cold pipeline. It is a better
noise-robust preparation circuit obtained through improved noiseless fitting
and compression. The next most informative control is **batch-only training
with the same 20-step rate-0.5 prefix followed by rate 0.2**, retaining all
other construction/pruning settings. That separates an aggressive step-size
schedule from the online update-order contribution. This control has not
been run as part of the present fixed grid.

Further confirmation also needs fresh target states and more seeds, and a
hybrid-trained direct compact-depth comparator before attributing the hybrid
pipeline's advantage specifically to pruning. All settings in this screen
reuse development targets; choosing H20/rate-0.5 after comparing nine settings
does not make it a held-out winner. The screen does not test real hardware,
other noise models, larger qubit counts, external optimizers, or state-of-the-art
competitors, and supports no general superiority claim yet.
