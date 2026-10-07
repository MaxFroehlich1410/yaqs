# Retained forward MPS tensor storage

`mps_memory.forward_tape_storage(trajectories, target=None)` measures the live public trajectories returned by `noisy_state_preparation_contribution`. It reads both `trajectory.states` (initial state and post-gate/post-noise states) and `trajectory.gate_outputs` (pre-noise gate outputs). It neither propagates a state nor modifies the supplied arrays. The standalone helper introduces no scaling experiment, numerical approximation, or authorization to load q8 targets.

The report distinguishes three quantities:

- Logical tensor bytes sum `ndarray.nbytes` at every retained tensor-reference occurrence. Repeated references appear repeatedly here.
- Unique-array logical bytes count each ndarray object once. Different views remain distinct objects even when they reference the same allocation.
- Known NumPy-owned backing payload bytes follow array base chains and count each owning ndarray payload once. A small view can retain a much larger owner. Disjoint views of one owner still share one retained allocation. Equal values in independent arrays never merge.

Base traversal covers ordinary views, negative/noncontiguous strides, NumPy stride-wrapper objects, and memoryview references to NumPy owners. Foreign-backed arrays retain logical byte measurements and an explicit unknown complete backing total. Python object arrays are rejected because their referents fall outside numeric tensor payload storage. NumPy owner payload bytes exclude allocator padding and are not an operating-system allocation measurement.

Separate `states`, `gate_outputs`, `forward_tape`, `exact_target`, and `retained_union` summaries use the same buffer registry. Shared target/tape backing is counted once in the union. `target_incremental_known_numpy_payload_bytes` counts only target owners absent from the tape; it is explicitly a known subtotal when foreign buffers occur. An absent target is labeled separately from an observed target with zero elements. Exactness of the supplied target remains the caller's responsibility; the utility performs no target factorization or fidelity check.

This is a snapshot of simultaneously retained tensor storage. It excludes noise-map and gate matrices, Python containers, allocator overhead, unreturned adjoints and derivatives, transient contractions, target factorization, and process RSS. It is not a peak-memory measurement and cannot establish scaling of the complete structural candidate. Summing tapes from separate sequential calls would require actually retaining those tapes together; the utility cannot reconstruct their past simultaneous lifetime.

The JSON-compatible output contains local buffer ordinals, owner types, and byte totals, but no memory addresses or retained array references. Synthetic tests cover q1-q3 actual public tapes, repeated states and arrays, disjoint shared views, negative strides, stride wrappers, memoryviews, independent equal-valued arrays, shared targets, foreign buffers, and empty tapes.
