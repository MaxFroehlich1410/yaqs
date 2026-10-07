# Parent-bound confirmation construction support

This is preliminary software support, not a confirmation freeze or execution
authorization. No held-out target is loaded, constructed, optimized or scored
while developing this wrapper. There is no CLI and no target loader. Its caller
must pass the normalized complex128 vector explicitly. Small synthetic q1-q3
engineering tests are separate from the scientific path. There is no scaling
support in this component.

## Unchanged numerical recipe

Call the already tested, frozen `construction.run_pipeline` directly. Do not
patch a module global, alter `construction.Config`, call the newer trajectory
comparison constructor, reuse archived parameters as a free state, or change
the underlying MPS implementation. The scientific configuration is exactly
`construction.Config()` with mode `development`: that name identifies the
validated recipe, while this wrapper's outer scientific phase is `confirmation`.

The seed root is 2026090520 with the original pilot initialization and training
namespaces. Preserve physical preparation from zero, layer depths 1-4, 100
updates per layer (25 online at 0.7, then 75 batch at 0.2), six magnitude-pruning
rounds with 20 batch repairs at 0.1, then 50 final batch updates at 0.1. All details,
initial angle scales, gate origins, pruning ties and protected gates remain in
the frozen [CONSTRUCTION_PROTOCOL.md](CONSTRUCTION_PROTOCOL.md). The pipeline
itself asserts 570 updates, 78 one-qubit gates, 30 two-qubit gates and 108 parameters.
All full gate-noise flags remain intact. No best checkpoint is selected.

MPS truncation remains max_bond_dim=None, svd_threshold=0, min_bond_dim=1 and
discarded-weight mode. `constructor_spec()` includes this exact configuration,
seed, physical endpoint counts, legacy pipeline version, this wrapper/protocol
and all construction numerical source hashes. The future parent must store
that exact object and include every constructor source in its own complete
algorithm source manifest. Changes require a new tested development recipe.

## Required parent schema

The parent is a `pilot.write_record` checksummed JSON artifact. The caller
supplies both its absolute path and its canonical content SHA-256; the wrapper
does not infer acceptance from a filename, a `--confirmation` flag, or a draft.
`parent_protocol(path, expected_checksum)` requires all of these fields:

- `schema`: `overnight.complete_confirmation_parent.v1`.
- `version`: a nonempty identity for the complete frozen experiment.
- `status`: `frozen`; `phase`: `confirmation`.
- `targets`: exactly, in order, `tfim_ferro`, `tfim_para`, `haar_random_2`,
  `haar_random_3`, `random_mps_bond3`.
- `replicas`: an explicit sorted nonempty unique subset of 11 and 29. This
  support component does not choose the eventual replication budget.
- `cases`: the complete target-by-replica product. Each row has exactly
  `target_id`, `replica`, `num_qubits` (6), and `target_vector_sha256` (SHA-256 of
  the contiguous complex128 dense vector bytes). Replicas of the same target
  must have the same vector hash. The parent loader is responsible for binding
  these hashes to the unchanged benchmark fixture; the wrapper never opens it
  to select or load a target.
- `algorithm`: nonempty structured `initializer`, `compiler`, `training`,
  `structure`, and `endpoint` sections describing the complete selected recipe.
  Each section must be a nonempty JSON object, even for an explicitly disabled
  component. Scientific completeness remains the coordinator's responsibility;
  this validator cannot infer whether an arbitrary algorithm description is
  scientifically sufficient.
- `arms`: at least two distinct named arms, each containing a nonempty list
  `roles` and a nonempty `configuration` object. Shared numerical arms may have
  multiple roles rather than duplicated executions.
- `budgets`: exactly the same arm keys, each with a nonempty object specifying
  all its frozen work caps. The parent binds optimizer, estimator, structural,
  compiler and stopping budgets; this wrapper introduces no tuning choice.
- `acceptance`: named `candidate` and distinct `primary_baseline` arm keys,
  a nonempty `primary_metric`, and an explicit `rule` object. The rule requires
  `route` (`fidelity` or `resource`), a nonempty list of testable textual
  `criteria`, `complete_panel_required: true`, and
  `failed_case_policy: retain_and_fail`. The final thresholds and statistical
  or deterministic acceptance criteria must be chosen from development only.
- `seed_policy`: a nonempty explicit mapping covering the parent's random
  domains and pairing. Construction itself always retains its exact legacy seed.
- `constructor`: exactly `constructor_spec()`.
- `source_hashes`: the complete algorithm's relative repository-path-to-SHA-256
  mapping, including all constructor hashes. Every file must exist and still
  match; paths outside the repository are rejected.
- `benchmark_hashes`: exactly the overnight startup custody hashes.
- `environment`: exactly `construction.environment()` (Python/dependencies and
  numerical thread environment).

Additional parent fields are allowed and remain inside its complete hash. No
fields may be changed after a case is bound. The whole parent hash, supplied
target-vector hash, target/replica identity, recipe, initialization seed,
environment and sources bind each child `PROTOCOL.json`. A changed parent,
source file, seed, target vector, arm choice, budget or acceptance rule fails
before construction or before returning a cached endpoint.

## API and durable execution

The future parent calls `prepare(root, vector, target_id, replica,
parent_path=..., parent_checksum=...)` only after its own complete freeze.
Preparation verifies the parent first, validates the supplied finite normalized
q6 vector against the selected parent case, and writes a child manifest without
performing quantum evolution. No scientific preparation is part of this support
implementation task. A nonempty unrelated output directory cannot be adopted.

Then call `run(root, manifest, vector)`. It revalidates everything and invokes
the frozen pipeline. Exact current theta, full scalar trace, phase counters,
chunk counts and intermediate gate origins remain in the unchanged legacy
`blocks/*.json`. The wrapper records every current block checksum after each
catchable attempt and verifies it before resume; all final block digests remain
in the endpoint. Block-level signatures also bind the exact circuit, initial
parameters, target bytes, schedule, optimizer seed and backend.

Each attempt is durably marked running before numerical work. Catchable errors
and interrupts retain measured process CPU, wall time, failure text and block
hashes; the same state resumes without reseeding or changing the schedule.
All repeated construction orchestration, magnitude decisions, checkpoint I/O,
replayed unfinished chunks and final MPS diagnostics are charged. The reported
CPU/wall includes successful invocation preflight, supplied-vector copies and
source/parent custody checks; prepare-only administration, read-only completed
cache checks, terminal ledger serialization and final result serialization are
outside that interval. The caller separately charges loading the supplied
target. Existing historical CPU is never substituted for this fresh cost.

A hard kill leaves a running/null-cost marker. Unknown, missing, nonfinite or
negative durations **block further numerical resume** until externally resolved
accounting is supplied and audited. Existing blocks/results without their
attempt ledger also fail closed. This wrapper offers no tool to erase, reset or
impute such costs. Caught failures remain in the ledger even after a successful
resume. The parent must retain them in its final scientific accounting.

Engineering calls use an explicit reduced `construction.Config(mode="engineering",
...)`, normalized synthetic q1-q3 vectors and `synthetic_` identities, with no
parent. Tests compare its exact circuit, theta, cold endpoint, pruning choices
and scalar block traces to the frozen constructor, including interruption
inside layer construction, pruning repair and final refinement. Metadata-only
q6 parent fixtures are artificial and never invoke a q6 quantum kernel.
