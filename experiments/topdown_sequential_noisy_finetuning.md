# Noisy cross refinement after sequential ideal pruning

Start all nine six-qubit targets and seeds 300/400/500 from the authenticated
`sequential_ideal` checkpoints in the completed pruning screen. Preserve their
gate schedules and retained angles exactly as inputs. Perform no construction,
pruning or depth reselection.

Run **100 online noisy sweeps followed by 200 batch noisy updates** (300 total).
Both phases use the unchanged archived cross-trajectory update, rate 0.2 and
zero decay. Use the same fixed 100-history IBM sampler-law bank, with seed
equal to the original optimizer seed plus 5000. Compress duplicate no-jump
histories with exact multiplicity weights. Authenticate the uncompressed bank
hash across the stage transition. If enabled noise yields no sampled jumps,
fail that case without resampling or falling back to noiseless optimization.

Seal inputs, the online stage, the batch stage, and assessment independently.
Reconstruct the bank with the same deterministic counter on resume, verifying
that both stages use the identical bank. Splitting the stages must reproduce
the parameters and fidelity trace of one uninterrupted archived optimizer call
with `steps=300, online=100`. Keep the final iterate; exact test scores never
select checkpoints. Evaluate initial, online and final noisy endpoints with
independent Qiskit density evolution and verify compiled-state equivalence.

Launch all 27 cases in parallel on Leonhard-13/16 after source/input checks,
targeted tests and observed capacity checks. Each numerical worker uses one
thread, at most 8 GiB and two hours. Unique launch claims prevent duplicate
submissions. Check completion every five minutes and stop monitoring after
audited completion or a terminal failure.

Report all targets and seeds, exact and sampled fidelities, phase budgets,
resources, costs and historical B comparison. The old B recipe used 100 online
plus 100 batch updates. This experiment changes pruning and batch budget;
its historical comparison does not isolate the pruning effect after training.

Use `--action prepare --parent <pruning-screen-root> --output <fresh-root>`.
Launch each case with `--action run --input <fresh-root>/inputs/<case>
--output <fresh-root>/jobs/<case>`. Original numerical source and production
defaults stay unchanged.
