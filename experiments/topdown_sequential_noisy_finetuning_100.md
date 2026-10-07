# Sequential pruning with 100 online and 100 batch updates

Run all nine six-qubit targets and seeds 300/400/500 from the same authenticated
sequential-ideal checkpoints as the completed 100-online/200-batch experiment.
Keep the original driver and all numerical sources unchanged. The separate
`experiments.topdown_sequential_noisy_finetuning_100` entry point changes only
the batch budget to 100 and the final-iterate identity to 200 total updates.
Each worker runs in its own process and restores the imported default on exit.

Both phases use rate 0.2, zero decay and one fixed weighted 100-history IBM noise
bank with training seed equal to optimizer seed plus 5000. Authenticate the bank
and multiplicities across phases. Freshly run both phases to measure CPU; do not
reuse cached online timing. No new construction, pruning, depth selection,
parameter repair or BFGS is performed. Charge the previously measured sequential
pruning cost separately. Preserve the final iterate without test-based selection.

Use a fresh root `output/q6-sequential-noisy-finetuning-100-20261006`. Prepare with
`--action prepare --parent output/q6-pruning-screen-20261006 --output <root>`.
Run each case with `--action run --input <root>/inputs/<case> --output <root>/jobs/<case>`.
Deploy the sealed source/input bundle to shared NFS at
`/Home/varioper/froehlich/yaqs-checks/q6-sequential-noisy-finetuning-100-20261006`.
Use the existing interpreter
`/Home/varioper/froehlich/yaqs-checks/q12-20260927-batches/.venv/bin/python`.
Split the frozen 27 cases into 14/13 assignments on Leonhard-13/16 and launch
only after source/input authentication, 60 passing targeted tests on both hosts,
capacity checks and unique per-case claims. Each worker uses one numerical
thread, an 8 GiB RSS guard and a two-hour maximum. Existing claims forbid
duplicate submission. Standard SSH options are BatchMode=yes, ConnectTimeout=30,
ClearAllForwardings=yes and StrictHostKeyChecking=yes. Leonhard-16 additionally
uses `output/q12-remote-checks-20260927/cluster_known_hosts` locally.

Collect online/batch/final stages and process/validation/launch/capacity receipts.
Independently replay initial, online, final noisy and final ideal fidelities.
Require online parameters and training traces to match the completed longer run,
and require the 100 batch traces to equal its first 100 batch traces. This directly
measures the fidelity gain from batch updates 101–200 on unchanged checkpoints.

Report all three-seed means and per-case endpoints. Compare saved competitors
at seed 300 under the same logical IBM channel. CPU tables must distinguish
new fine-tuning, pruning plus fine-tuning, selected-circuit construction plus
pruning plus fine-tuning, and the complete actual checkpoint-reuse pipeline
including all earlier B depth-search candidates. Historical validation and test
costs stay separate. The last measure includes the old noisy B candidates that
selected the parent depth; it does not measure a fresh sequential-pruning depth
search. Competitor costs charge their full candidate grids. Timings from different
runs and implementations are descriptive, not an equal-hardware speed test.
