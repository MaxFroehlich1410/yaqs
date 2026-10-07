# Twelve-qubit Haar capacity and optimization audit

Two independent statevector branches start from the authenticated best depth-16
endpoint of `q12-statevector-continuation-20260929`: ideal fidelity
0.47015271601097486. The threshold is **0.999 ideal fidelity**, not rounded
display fidelity. Neither branch applies pruning or noise. Existing runs and
their numerical sources remain unchanged.

| Branch  | Change                        | Fixed settings                                                                                       | Limits                                                        |
| ------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `fixed` | More optimization at depth 16 | Original safeguarded L-BFGS, same parameters and topology                                            | First threshold hit or 10,000 additional updates              |
| `grow`  | Add depths 17 onward          | Original 100 online Krotov + 100 L-BFGS updates per depth; rate 0.7; seed 300; extension scale 0.001 | First successful depth, depth 128, or 24 additional CPU hours |

Fixed-depth optimization calls the original optimizer once with `online=0`.
The parent does not contain L-BFGS curvature pairs, so history resets once at
branch start and is preserved throughout that call. It does not repeatedly
restart the optimizer in small batches. The only backend instrumentation saves
the incoming accepted iterate every 100 gradient calls. These live snapshots
are observational and do not contain resumable optimizer history. If interrupted,
the fixed branch must replay its single call from the original parent, unless a
complete training checkpoint was sealed. Such a replay incurs additional cost
and must not be reported as uninterrupted training.

Growth calls `topdown_noiseless_continuation.construct` without changes and carries
its own preceding best parameters between depths. It never uses the result of
the independent fixed-depth branch. Every completed depth is sealed and reusable
after interruption. It retains the full 200-update per-depth budget even if the
threshold is first crossed inside that depth, recording the first crossing index.
The production three-depth patience rule does not stop this capacity audit;
stopping is driven by the requested fidelity threshold and explicit resource
limits. The CPU ceiling is checked between completed depths and can be exceeded
by at most one depth. An external supervisor provides a 30-hour wall and 16-GiB
memory guard per branch; interruption remains an incomplete result.

Success means the threshold was observed at the recorded depth and update count.
It does not establish the minimum required depth, optimizer budget, or gate count.
Exhaustion does not prove the target unreachable or the optimizer converged.
Compare achieved fidelity versus additional CPU time, including the historical
parent cost separately. Parent target, source, topology, parameters, and selected
fidelity are authenticated before either branch starts; endpoint scores are
replayed with the exact statevector backend.

```sh
python -m experiments.topdown_haar_capacity \
  --parent /path/to/q12-statevector-continuation-20260929/results/haar_random_1 \
  --output /path/to/new/results/fixed --arm fixed

python -m experiments.topdown_haar_capacity \
  --parent /path/to/q12-statevector-continuation-20260929/results/haar_random_1 \
  --output /path/to/new/results/grow --arm grow
```

Run independent branches on available Leonhard hosts, with one numerical thread
and low scheduling priority, without altering existing experiments. Record exact
commands, capacity observations, frozen source hashes, logs, and supervisor exit
receipts. Progress is in each branch's `progress.json`; final results are in
`final/record.json`. Fixed-depth fidelity curves, parameters and complete optimizer
records are in `training/arrays.npz`; optimizer records use a compressed uint8
UTF-8 JSON payload, with references in the record. Growth checkpoints retain the
original `record.json`, `arrays.npz`, and checksum `complete.json` format.
