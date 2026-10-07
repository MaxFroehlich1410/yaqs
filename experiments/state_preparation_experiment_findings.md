# Noisy State-Preparation Experiment Findings

Shared-adjoint refinement follow-up (2026-09-12): all 7040 deletion scores and
all search decisions reproduce the exhaustive implementation. Reinvesting the
saved CPU raises mean noisy fidelity to 80.955% for adaptive online Krotov and
80.839% for Adam, versus 79.801% for compute-matched fixed Adam. Both pass the
development screen; Krotov's +0.116 pp lead over structural Adam comes from
only the two Haar pairs, while Adam wins all six structured pairs. See the
[shared-adjoint findings](state_preparation_adjoint_restructuring_findings.md).

Noisy restructuring follow-up (2026-09-12): the eight-case, 32-arm exact-score
deletion-and-repair screen improves the three structured families by roughly
1.3-1.7 pp over compute-matched fixed Adam, but loses on Haar. Adaptive online
reaches the higher structural endpoint mean, without a consistent matched-CPU
advantage; neither optimizer meets the +0.5 pp overall screen. Exhaustive
deletion scoring consumes 85.2% of structural CPU. See the
[noisy restructuring findings](state_preparation_noisy_restructuring_findings.md).

Adaptive-online follow-up (2026-09-12): the matched 64-run screen is complete.
Adaptive scaling improves online Krotov in all eight pairs, but every adaptive
rate still loses to Adam 0.02 at both the endpoint and matched training CPU.
See the [adaptive-online findings](state_preparation_adaptive_online_findings.md).

Weighted optimization follow-up (2026-09-12): the matched 96-run screen is
complete. Weighted sampling improves Adam more than Krotov; weighted Adam wins
all eight pairs against each Krotov schedule at R32 and R128, with lower CPU
cost. See the [weighted optimization findings](state_preparation_weighted_optimization_findings.md).

Sampling-budget follow-up (2026-09-12): Adam remains ahead in the matched
R=32/R=128 screen. A subsequent weighted no-factor diagnostic substantially
improves gradient error per CPU second. Weighted optimizer training had not
yet been run at that stage; the follow-up above now covers it. See the
[sampling-budget findings](state_preparation_sampling_budget_findings.md).

Statevector follow-up (2026-09-11): the four-family gradient/cross, tail-rate and
CRN development screens are complete. All-online gave a small consistent
endpoint gain; R=32 endpoint gradients remained noisy, so the conditional
adaptive screen was deferred. See the
[new findings](state_preparation_trajectory_diagnostics_findings.md).

Calibration follow-up (2026-09-05): the original q6 causal and hybrid results
used higher per-gate noise than the intended hardware-error targets. Preserve
the historical findings below, but interpret those runs as higher-noise tests.
The controlled corrected-noise reruns and independent channel checks are
documented in [the calibration follow-up](state_preparation_calibrated_rerun_findings.md).

- **Evidence review date:** 2026-09-04
- **Current branch:** `codex/minimal-noisy-krotov`
- **Latest experiment implementation commit:** `8f8308ad5c3dfb9aed0430c92464f6a1f416608f`

## Purpose and scope

This document summarizes the state-preparation experiments retained in the
current repository or recoverable from its Git history. Its purpose is to guide
the research program toward a genuinely new method for quantum state
preparation in the presence of noise. It is an evidence review, not a claim
that the current method is already state of the art.

The review covers:

- the current benchmark definition and minimal experiment runner;
- numerical CSV traces, figures, and their generating scripts before the
  repository was minimized;
- the five-target legacy evidence audit and the experiment handoff;
- claims in the current manuscript; and
- whether a result used fresh noisy evaluation, fixed training trajectories,
  or only a training trace.

Unrelated quantum-dynamics, process-tensor, supervised-classification, and
solver-stress experiments were not treated as evidence for noisy state
preparation. Tests of experiment infrastructure establish that code paths work;
they are not numerical evidence for an algorithmic advantage.

## Executive conclusions

1. **Noise-aware state-preparation optimization can run and learn in small or
   weak-noise settings.** The strongest clean diagnostic is the four-qubit
   experiment at noise strength `0.002`, where fixed-validation fidelity rose
   from roughly `0.034` to `0.636` with the cross-trajectory update and from
   `0.043` to `0.574` with the independent update. This is a single target with
   unequal trajectory budgets, so it establishes feasibility rather than
   superiority.

2. **Noiseless or weak-noise layerwise warm starts are consistently more useful
   than deep from-scratch optimization.** An eight-qubit Gaussian target reached
   `0.9909` noiseless fidelity from a target-informed initialization. Five
   eight-qubit dense random targets reached final fixed-trajectory training
   fidelities of `0.8924` to `0.9620` after depth growth. Conversely, independent
   depth-4/8/12/16 runs from scratch ended at `0.3476`, `0.2670`, `0.2011`, and
   `0.0806`, respectively. The latter result diagnoses an optimization problem;
   it does not prove that deeper circuits are less expressive.

3. **Common random numbers (CRN) stabilize a sampled training objective, but the
   repository does not show that permanently frozen trajectories improve the
   true noisy objective.** In the retained CRN comparison, fixed CRN produced a
   monotone trace from `0.0976` to `0.1424`. The resampled independent and cross
   variants were highly nonmonotone but both ended near `0.202`, above the CRN
   endpoint. The manuscript claim of a roughly `60%` unstabilized cap and CRN
   convergence above `91%` is contradicted by the CSV associated with its
   figure.

4. **The strongest archived five-target result belongs to a predetermined
   layerwise BMPD procedure, not ADAPT-VQE.** Under the shared 500-trajectory
   simulated-noise evaluation, that method obtained mean fidelity `0.7905`,
   versus `0.7138` for fixed-depth VQA and `0.5961` for the top-down method. It
   beat fixed-depth VQA on all five targets by a mean paired difference of
   `0.0766`. Its code grows a fixed brickwall ansatz through depths 1, 2, 3, and
   4; it performs no operator-pool selection and must not be called ADAPT-VQE.

5. **The retained five-target data do not validate the proposed iterative
   impact-pruning contribution.** The reported top-down rows used one-shot angle
   magnitude pruning from 1,032 to 150 parameters. They did not use the separate
   impact score implementation. The method beat fixed-depth VQA on four targets
   by small margins, failed on the fifth, never beat the layerwise method, and
   had a lower aggregate mean. The cause of the fifth failure is unknown because
   stage and gradient traces were not retained.

6. **Noise strength is currently a more decisive factor than the choice between
   the independent and cross update formulas.** Across six four-qubit target
   seeds, mean final fixed-validation fidelity fell from `0.7181` (independent)
   and `0.6668` (cross) at `gamma=0.001` to `0.1950` and `0.1834` at
   `gamma=0.012`, and to about `0.09` or below at most larger strengths. The two
   methods used different trajectory budgets, so this sweep cannot establish a
   fair winner.

7. **The first canonical causal q6 experiment does not support refresh-5 or a
   general noisy-fine-tuning advantage.** A shared layerwise checkpoint reached
   `0.971599` noiseless fidelity. Across three independent noisy-training
   replicates and 500 fresh paired test trajectories per replicate, refresh-5
   changed fidelity by `-0.000916` relative to the frozen checkpoint (replicate-
   level 95% Student-t interval `[-0.003312, 0.001480]`). Per-update resampling
   changed it by `+0.002348` in the same comparison and was positive in all
   three replicates, but its interval `[-0.000850, 0.005545]` still included
   zero. Refresh-5 lost to resampling in all three replicates by a mean
   `-0.003264`, with interval `[-0.005204, -0.001323]`.

8. **The repository does not yet support a state-of-the-art claim.** There is no
   completed canonical 6/12-qubit benchmark matrix, no real-QPU execution, no
   correctly identified external state-preparation competitor, no fair
   normalized-work comparison, and only one canonical target in the new
   multi-seed held-out study.

## Evidence standard used in this review

The artifacts have different scientific weight:

| Evidence class      | Meaning                                                                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Comparative         | Multiple methods evaluated on the same fresh trajectory protocol. This is the strongest retained evidence, although raw trajectories and full provenance may still be missing. |
| Diagnostic          | A numerical trace or fixed validation ensemble useful for learning about optimizer behavior, but not an unbiased final comparison.                                             |
| Figure/console only | A script or figure exists without a numerical result table sufficient to recompute the claim.                                                                                  |
| Protocol only       | Runnable code and tests exist, but there is no completed scientific result.                                                                                                    |

None of the archived artifacts qualifies as confirmatory evidence. The new q6
study is a well-controlled exploratory comparison with complete raw
trajectories, but it covers only one target and three optimization replicates.
In particular, a smooth fixed-CRN training trace measures optimization of one
finite frozen ensemble, not generalization to the noisy channel expectation.

## Findings by experiment family

### 1. Noiseless Gaussian SMPD state preparation

Historical artifact:
`experiments/krotov_gaussian_smpd_state_preparation_trace.csv` at commit
`6bdc86baeb7daa03756f153b49fda0d7606d04d5`.

| Setting          |                                                      Value |
| ---------------- | ---------------------------------------------------------: |
| Qubits           |                                                          8 |
| Ansatz           | Three-layer right-sweep SMPD-style circuit, 222 parameters |
| Updates          |                                          400 batch updates |
| Initial fidelity |                                                 `0.015923` |
| Final fidelity   |                                                 `0.990877` |

**Interpretation.** The noiseless optimizer can prepare this structured target
to high accuracy when supplied with a target-informed product-state warm start.
This is strong evidence for the value of pretraining and initialization, but it
is not a generic from-scratch result. The historical script used
`numpy.linspace(0, 1, 256)`, including the endpoint; the current normative
Gaussian target uses an endpoint-excluded quantics grid. The numerical result
therefore cannot be cited as a result for the current target fixture.

### 2. Noisy Gaussian fine-tuning

Historical artifacts at commit `6bdc86b`:

| Noise/update              | Initial trace mean | Final trace mean | Maximum trace mean |
| ------------------------- | -----------------: | ---------------: | -----------------: |
| `gamma=0.02`, independent |         `0.187694` |       `0.247441` |         `0.268191` |
| `gamma=0.02`, cross       |         `0.187694` |       `0.247760` |         `0.268774` |
| `gamma=0.002`, cross      |         `0.845824` |       `0.865212` |         `0.936415` |

All three runs used 50 trajectories and 50 noisy updates from the high-fidelity
noiseless warm start. Approximately half of the successive reported means
decreased. The independent and cross results at `gamma=0.02` were practically
identical.

**Interpretation.** The weak-noise run retained high mean fidelity, whereas the
ten-times-stronger noise reduced it severely. However, each trace point used a
new sampled ensemble, so differences between its initial, maximum, and final
means mix parameter improvement with Monte Carlo variation. There is no paired
fresh pre/post test. These traces motivate noise-strength calibration but do not
show that fine-tuning improved held-out performance.

### 3. Four-qubit feasibility and noise-strength sweep

The 100-update, `gamma=0.002` diagnostic used a reachable three-layer,
59-parameter target and monitored a fixed validation ensemble disjoint from the
resampled training ensembles.

| Method            | Final fixed noisy fidelity | Final noiseless fidelity | Training/evaluation trajectories |
| ----------------- | -------------------------: | -----------------------: | -------------------------------: |
| Noiseless         |             not applicable |               `0.870888` |                            `0/0` |
| Noisy independent |                 `0.573698` |               `0.816284` |                          `16/32` |
| Noisy cross       |                 `0.635773` |               `0.814925` |                           `4/12` |

The broader 80-update sweep covered target seeds 123 through 128. Selected
means over those six targets are:

| `gamma` | Independent mean | Cross mean | Cross wins |
| ------: | ---------------: | ---------: | ---------: |
| `0.001` |       `0.718139` | `0.666753` |        2/6 |
| `0.012` |       `0.194999` | `0.183449` |        2/6 |
| `0.023` |       `0.091411` | `0.077717` |        2/6 |
| `0.045` |       `0.070876` | `0.080744` |        5/6 |
| `0.100` |       `0.074483` | `0.060787` |        2/6 |

**Interpretation.** The optimizer learns under weak noise, but performance
collapses rapidly as the per-operator rate increases. Cross updates are not a
consistent winner across targets or strengths. Because the cross runs used
4 training and 12 validation trajectories while independent runs used 16 and
32, the comparison confounds update rule, compute, and estimator variance.
Repeatedly observing the same validation ensemble also makes it unsuitable as
an untouched final test set.

### 4. Eight-qubit scaling and depth

Several from-scratch experiments showed that merely increasing circuit depth or
the number of iterations did not solve optimization:

| Experiment                         | Initial fidelity | Final fidelity | Main observation                                                    |
| ---------------------------------- | ---------------: | -------------: | ------------------------------------------------------------------- |
| Depth-4, strong-noise, 50 updates  |       `0.004008` |     `0.004148` | Monotone but effectively no learning.                               |
| Deeper/large-step run, 500 updates |       `0.007435` |     `0.223794` | Better, still low.                                                  |
| Best 30-update step-size search    |       `0.007435` |     `0.107237` | Large batch step was best of four tested settings.                  |
| Noisy layerwise depth 4 to 16      |       `0.003794` |     `0.690783` | Warm-started growth substantially improved the fixed-CRN objective. |

Independent noiseless runs from scratch provide an especially clear warning:

| Depth | Final fidelity | Best fidelity during run |
| ----: | -------------: | -----------------------: |
|     4 |     `0.347627` |               `0.347627` |
|     8 |     `0.266985` |               `0.353073` |
|    12 |     `0.201054` |               `0.317637` |
|    16 |     `0.080564` |               `0.233037` |

**Interpretation.** The limiting factor in these runs was trainability and
step-size control, not necessarily ansatz capacity. Depth must be introduced
through a continuation or layerwise schedule, with checkpoint selection, rather
than assumed to help from random initialization.

### 5. Five dense random targets with layerwise fixed-CRN training

The historical `layerwise_8q_5states.csv` grew circuits through depths
4, 8, 12, 16, 20, and 24, with 250 updates per depth. It used one frozen
X-noise trajectory at strength `0.0005`.

| Target seed | Final training fidelity | Maximum training fidelity |
| ----------: | ----------------------: | ------------------------: |
|          42 |              `0.892370` |                `0.902503` |
|         101 |              `0.945510` |                `0.945510` |
|        2024 |              `0.958707` |                `0.958707` |
|         777 |              `0.928888` |                `0.930807` |
|         999 |              `0.962032` |                `0.962032` |

**Interpretation.** Layerwise continuation can fit a deep circuit to several
eight-qubit targets. This is useful algorithm-development evidence. It is not a
channel-level noisy-fidelity result: one frozen trajectory is both an extremely
small sample and the optimized objective, and no fresh evaluation was retained.
The targets also differ from the current benchmark's fixed Haar targets.

### 6. CRN ablation

The CSV used by the manuscript's CRN figure contains 20 updates:

| Condition                    |    Initial |      Final | Decreasing steps |
| ---------------------------- | ---------: | ---------: | ---------------: |
| Noiseless                    | `0.027302` | `0.058511` |             0/20 |
| Noisy independent, resampled | `0.084277` | `0.202643` |            11/20 |
| Noisy cross, resampled       | `0.084277` | `0.202058` |            11/20 |
| Noisy cross, fixed CRN       | `0.097559` | `0.142386` |             0/20 |

**What worked.** Reusing the random maps converted this sampled objective into
a smooth, monotone optimization path.

**What did not work.** The fixed-CRN endpoint was lower than both resampled
endpoints, and all four conditions started from different sampled objective
values. No common held-out ensemble was used to rank the final parameter
vectors. The experiment therefore supports a variance/stability statement, not
a fidelity or generalization advantage.

Refreshing every five updates was a sensible response to this failure mode, but
the canonical paired q6 experiment reported below did not validate it. In that
experiment, refresh-5 underperformed per-update resampling in every replicate.

### 7. Five disordered-TFIM method comparison

The most informative archived comparison is
`experiments/results/rigorous_benchmark_5states.csv` at `6bdc86b`. It used five
eight-qubit disordered-TFIM targets and 500 final TJM trajectories under a
logical, IBM-inspired noise model.

| Archived label  | Correct identity                                                                              | Mean noiseless fidelity | Mean noisy fidelity |              Noisy range |
| --------------- | --------------------------------------------------------------------------------------------- | ----------------------: | ------------------: | -----------------------: |
| Standard VQA    | Fixed depth-4 BMPD, direct noisy training                                                     |              `0.899071` |          `0.713846` | `0.638677` to `0.747143` |
| ADAPT-VQE       | Predetermined layerwise depth 1 to 4 BMPD, then noisy fine-tuning                             |              `0.981470` |          `0.790494` | `0.748201` to `0.813334` |
| Top-Down Krotov | Layerwise depth 4 to 16, one-shot magnitude pruning to 150 parameters, then noisy fine-tuning |              `0.690263` |          `0.596128` | `0.027595` to `0.788881` |

The layerwise method beat fixed-depth VQA on every target. The top-down method
beat fixed-depth VQA on four targets, but only by `0.0035` to `0.0546`, and its
fifth-target failure reduced the aggregate mean. Excluding that failure gives
the top-down method a mean of `0.738261`, still below the corresponding
four-target layerwise mean of `0.786062`. Top-down never beat the layerwise
method on an individual target.

This table is the strongest retained evidence because all methods went through
the same nominal 500-trajectory evaluation. It nevertheless has serious limits:

- the simulations were not independently reproduced;
- final parameters, raw trajectories, uncertainty, full traces, invocation
  logs, and runtime fingerprints were not retained;
- the evaluation used deterministic default seeds 0 through 499, not unseeded
  trajectories;
- the noise was a simulated hardware-inspired logical Lindblad profile, not an
  IBM Heron QPU execution;
- optimization budgets were unequal: 300 updates for fixed VQA, 400 noiseless
  plus 200 noisy updates for the layerwise method, and 1,000 pretraining plus
  300 noisy updates for top-down; and
- the legacy targets are q8 disordered-TFIM states, whereas the current
  benchmark fixes q6/q12 uniform-TFIM and other target families.

Most importantly, this comparison does not contain a paired evaluation of each
layerwise circuit immediately before and after noisy fine-tuning. It cannot
isolate whether the final advantage came from noiseless layerwise pretraining,
the larger optimization budget, or the noisy update stage.

### 8. Pruning studies

The repository history contains magnitude-pruning sweeps, a separate iterative
impact-pruning script, and pruning figures. The numerical five-target CSV was
generated by magnitude pruning. The impact-pruning script printed results but
did not retain a machine-readable result table, checkpoints, or fresh
evaluation samples.

The earlier `hardware_benchmark_5states.csv` is only a three-fixed-trajectory
training-map comparison. Its high pruned values on several targets and its
failure near zero on another are useful for diagnosing instability, but they
are not hardware fidelities. The identical IBM/Google values for some rows are
also consistent with tiny frozen ensembles containing no differentiating jump
events.

**Interpretation.** Compression to 150 parameters is implemented, and magnitude
pruning sometimes preserves a useful circuit. The repository does not
demonstrate that impact pruning is better than magnitude pruning, that 150 is
optimal, or that the compression improves fresh noisy fidelity at equal work.
This proposed contribution must be treated as unvalidated.

### 9. Current canonical benchmark and first causal q6 study

The current [benchmark definition](../benchmarks/state_preparation_benchmarks.md)
fixes 18 q6/q12 target instances, exact target fixtures, six minimum or twelve
full noise conditions, disjoint seeds, fresh test trajectories, uncertainty,
and circuit/work reporting. The minimal branch can run its reference
noiseless-train/noisy-evaluate matrix. No completed full-matrix `results.jsonl`
or derived result table is present in the repository.

The current [refresh runner](noisy_krotov_refresh_minimal.py) defines one q8,
depth-4, seed-100 exploratory experiment:

- start from a noiseless pretrained checkpoint;
- fine-tune with three trajectories;
- refresh the frozen ensemble every five updates;
- compare pretrained, historical fixed-three, and refreshed parameter vectors
  on the same 2,000 held-out trajectories; and
- keep training and evaluation random streams disjoint.

This is the right shape for testing whether noise-aware fine-tuning actually
works. At the original review date, neither of its required input checkpoints
nor a `refresh_study_result.json` existed in the repository. Commits `001692e4`
and `4fc7a0ff` likewise added calibration/refresh runners and tests without
committing completed outputs.

The new [canonical causal runner](noisy_krotov_causal_study.py), implemented at
commit `8f8308ad`, removes the missing-checkpoint problem by generating one q6
`tfim_critical` warm start through noiseless depths 1, 2, 3, and 4. It then
compares five arms from exactly that parameter vector: frozen, matched
noiseless continuation, fixed CRN, refresh-5 CRN, and per-update resampling.
Every noisy arm uses the same three-trajectory cross update. A disjoint fixed
12-trajectory stream selects checkpoints, followed by 500 fresh paired test
trajectories per replicate. Three noisy-training replicates were run.

The complete run finished successfully in `5,122.95` seconds. Its output has
823 files (`176 MiB` on disk), including all sampled maps, checkpoints, traces,
and 1,500 raw test rows. The result checksum is
`sha256:d339f9acc4a4bd556a18d64944a0777445891e404b20699e75cb180a9e4e6f7e`.
The seed audit found 1,800 unique training seeds, 36 validation seeds, and 1,500
test seeds, with the three domains pairwise disjoint. The full local artifact is
stored outside the Git repository at
`/Users/maximilianfrohlich/Desktop/yaqs-experiment-results/q6-tfim-critical-causal-8f8308ad/`.

Layerwise noiseless pretraining worked strongly:

| Depth | Initial noiseless fidelity | Final noiseless fidelity |
| ----: | -------------------------: | -----------------------: |
|     1 |                 `0.123435` |               `0.836756` |
|     2 |                 `0.836861` |               `0.911724` |
|     3 |                 `0.911736` |               `0.941236` |
|     4 |                 `0.941236` |               `0.971599` |

The primary held-out results use the noisy-training replicate as the sampling
unit:

| Selected arm           | Mean held-out fidelity | Difference from frozen | Replicate-level 95% Student-t interval |
| ---------------------- | ---------------------: | ---------------------: | -------------------------------------: |
| Frozen                 |             `0.259759` |             `0.000000` |                         not applicable |
| Noiseless continuation |             `0.259759` |             `0.000000` |                 `[0.000000, 0.000000]` |
| Fixed CRN              |             `0.261903` |            `+0.002144` |                `[-0.006099, 0.010387]` |
| Refresh-5 CRN          |             `0.258843` |            `-0.000916` |                `[-0.003312, 0.001480]` |
| Per-update resampling  |             `0.262107` |            `+0.002348` |                `[-0.000850, 0.005545]` |

No noise-aware arm reliably beat the frozen checkpoint at this sample size.
Per-update resampling improved over frozen in all three replicates, but the
replicate-level interval still crossed zero. Refresh-5 lost to resampling in all
three replicates: mean paired difference `-0.003264`, interval
`[-0.005204, -0.001323]`. Fixed CRN selected the unchanged checkpoint in one
replicate and had an uncertain positive aggregate effect.

The validation behavior is itself an important finding. For refresh-5,
validation selected iterations 200, 40, and 85 and suggested gains of roughly
`0.105`, `0.029`, and `0.017`. Those gains did not transfer to the fresh test
streams. Replicate 0 is the clearest example: validation rose from `0.286840`
to `0.391639`, but held-out fidelity changed from `0.271994` to `0.270453`.
Repeated checkpoint selection over only 12 fixed trajectories therefore
overfit the validation ensemble. Future studies need a larger selection stream,
fewer candidate checkpoints, or a more precise independent objective.

## Manuscript claims that must be corrected or rerun

The current [manuscript](../manuscript/main.tex) contains claims that are not
supported by the retained evidence:

| Manuscript claim                                                    | Repository evidence                                                                                                             |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| CRN converges above 91%, while unstabilized training caps near 60%. | Contradicted by `crn_comparison.csv`: final values are `0.1424` for fixed CRN and about `0.202` for both resampled variants.    |
| The bottom-up comparator is ADAPT-VQE.                              | Incorrect method identity: the script grows a predetermined BMPD ansatz and has no operator pool or adaptive selection.         |
| The five-target top-down rows use iterative impact pruning.         | Incorrect: they use one-shot angle magnitude pruning. Impact pruning is a separate script without retained five-target results. |
| Results are true IBM Heron hardware fidelities.                     | Incorrect: they are TJM simulations of a hardware-inspired logical noise model; no QPU jobs were run.                           |
| The 500 trajectories are unseeded.                                  | Incorrect: the library default deterministically used seeds 0 through 499.                                                      |
| The three algorithms were compared fairly.                          | Unsupported: update and trajectory-equivalent budgets differ substantially, and no complete work ledger was retained.           |
| A noise-induced barren plateau caused the fifth top-down failure.   | Unsupported causal diagnosis: the failure is real, but no gradient or stage trace was retained.                                 |
| Top-down achieves 72% to 79% on four of five targets.               | Numerically inaccurate: the nonfailed range is `0.6933` to `0.7889`.                                                            |

These corrections are not cosmetic. They change the identity of the leading
method, the nature of the execution, and the central numerical claim.

## What has worked, what has not, and what remains unknown

### Worked well enough to build on

- The Krotov state-preparation implementation passes focused numerical tests
  and has completed small noisy runs.
- Target-informed initialization and layerwise depth continuation can avoid
  severe from-scratch optimization failures.
- Weak-noise training can substantially increase a fixed validation objective.
- CRN provides short-term variance reduction and monotone optimization of a
  frozen sample-average objective.
- Fresh evaluation draws are technically feasible, and the five-target
  comparison used the same nominal final-evaluation protocol for every method.
- The current benchmark now defines stable targets, noise semantics, seed
  separation, uncertainty, and reporting much more carefully than the legacy
  experiments.
- Canonical q6 layerwise pretraining raised noiseless fidelity from `0.123435`
  at depth 1 initialization to `0.971599` at depth 4.
- The new causal runner retained and verified every map, checkpoint, raw paired
  trajectory, and disjoint random stream through a completed 85-minute run.

### Did not work or was not robust

- Deep circuits trained from scratch frequently stalled or regressed.
- Strong local Pauli rates collapsed useful fidelity in both q4 and q8 studies.
- Cross updates did not consistently outperform independent updates.
- Fixed CRN improved smoothness but not the retained ablation's final trace
  value, and it creates an overfitting risk.
- Magnitude-pruned top-down training suffered one catastrophic failure and had
  the worst five-target aggregate mean.
- Figure-only and console-only pruning results were not retained in a form that
  supports quantitative verification.
- The historical paper workflow attached algorithm, hardware, and causal labels
  that the scripts and artifacts do not justify.
- Refresh-5 failed to beat the frozen warm start on fresh q6 trajectories and
  underperformed per-update resampling in all three completed replicates.
- Selecting among 41 checkpoints on only 12 fixed validation trajectories
  produced large apparent improvements that did not generalize to the test
  stream.

### Still unknown

- Whether the small positive resampling effect persists with more than three
  optimization replicates and across targets.
- Whether periodic refresh can help at another cadence, trajectory count, or
  learning rate; refresh every five updates with three trajectories did not.
- Whether cross updates help at a matched trajectory-equivalent budget.
- Whether impact-based pruning beats magnitude pruning or an unpruned circuit
  under matched native gate counts and compute.
- Whether any advantage survives across the current 18 target instances,
  multiple initialization/optimization seeds, and all benchmark noise models.
- Whether the method beats correctly implemented literature baselines or
  hardware-native state-preparation methods.
- How results scale from q6 to q12 and from simulation to a real device.

## Research implication and recommended next decision

The completed q6 experiment changes the recommendation. Noiseless layerwise
pretraining remains the strongest component, but refresh every five updates
should not be treated as the leading noise-aware method. The current
three-trajectory update is too noisy, and the 12-trajectory selection ensemble
is too small relative to 41 candidate checkpoints.

The next experiment should be a locked q6 optimization diagnostic, not pruning
or a larger benchmark matrix:

1. retain the same shared layerwise checkpoint and frozen control;
2. focus on per-update resampling, with fixed CRN retained only as a diagnostic;
3. screen trajectory counts and learning rates on explicitly separate
   screening streams, recording update-normalized compute;
4. use substantially more validation trajectories and fewer checkpoint
   opportunities, so selection noise cannot dominate a `0.2%`-scale effect;
5. lock the chosen setting, then rerun it over more optimization seeds and at
   least several canonical q6 targets with one untouched paired test per run;
   and
6. only revisit refresh cadence if this diagnostic shows a reproducible
   bias-variance benefit over resampling.

Pruning should still wait until noisy fine-tuning reliably beats the same
unpruned frozen checkpoint. Otherwise compression would be layered onto a
noise-aware optimization effect that has not been established.

For a paper, the primary claim should initially be framed as a testable
hypothesis—variance-controlled noise-aware Krotov fine-tuning improves fresh
noisy fidelity—not as a state-of-the-art result. A state-of-the-art claim
requires the canonical q6/q12 target matrix, fair external baselines, uncertainty
over target and optimizer seeds, transparent failures, normalized work, and
preferably device validation.

## Repository and history map

| Commit                                     | Role in this review                                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `ad785ea5c4219633924ff646d09cd07976b9007a` | Added noisy Krotov, CRN, pruning, and the first experiment set.                                        |
| `fb621e2deb4da6f8ba16d3e48d05077d8e2b8809` | Added the manuscript and five-target evaluation artifacts later audited by Phase II.                   |
| `6bdc86baeb7daa03756f153b49fda0d7606d04d5` | Last convenient snapshot containing the legacy experiment scripts and numerical CSVs together.         |
| `1b986db84789d303eec30a4388bb0a2d27dd5a6b` | Handoff stating that the audited reproduction and corrected numerical study had not been completed.    |
| `001692e4363e4eba68fec72f2a4ed3597a94a290` | Added a q8 calibration runner, but no result artifact.                                                 |
| `4fc7a0ff7dadd36444ef46d532574e621a5912a5` | Added the trajectory-refresh study, but no result artifact.                                            |
| `bb26c50a22082af56cea08f2935dbd8fb9606de1` | Minimized the branch and removed the legacy experiment files while retaining the focused refresh path. |
| `1ff6620b9e43416d75771c573c1a3d6cda220f12` | Restored the concise canonical benchmark implementation and fixed target fixture.                      |
| `8f8308ad5c3dfb9aed0430c92464f6a1f416608f` | Added the canonical five-arm q6 causal runner used for the completed paired experiment.                |

The historical evidence can be inspected without restoring it into the working
tree. Examples:

```bash
git show 6bdc86b:experiments/crn_comparison.csv
git show 6bdc86b:experiments/results/rigorous_benchmark_5states.csv
git show 6bdc86b:experiments/re_evaluate_benchmarks.py
git show 1b986db8:benchmarks/state_preparation/phase2/data/legacy_evidence_audit_v1.json
git show 1b986db8:benchmarks/noisy_krotov_experiment_handoff.md
```

The historical audit is especially important: it recomputed the five-target
arithmetic but classified the claimed simulation reproduction as unreproducible
and identified the method-label, pruning-label, hardware-label, CRN,
evaluation-seed, causal, and budget discrepancies summarized above.
