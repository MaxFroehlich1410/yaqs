# Independent verification of the completed confirmation

The independent Qiskit oracle reproduced all **2,520 saved fidelity values**
within the frozen tolerance of $10^{-10}$. The largest absolute discrepancy
was $9.992007221626409\times10^{-15}$. Independent numerical verification passed;
the candidate's original performance promotion gate remains **failed**.

The complete comparison contains ten cases, seven methods, three saved panels
and twelve noise profiles. Its 210 role endpoints reduce to 141 unique
target/circuit/parameter combinations, requiring 1,692 independent profile
evaluations. All 2,520 comparison rows remain present. There were no failed
calls, missing rows or CPU-cap overruns.

| Check                                      | Observed bound                     | Frozen requirement   |
| ------------------------------------------ | ---------------------------------- | -------------------- |
| Maximum absolute fidelity discrepancy      | $9.992007221626409\times10^{-15}$  | At most $10^{-10}$   |
| Maximum density trace defect               | $1.3434149399528736\times10^{-14}$ | At most $10^{-10}$   |
| Maximum reported Hermitian residual        | $3.9101250860578366\times10^{-16}$ | At most $10^{-10}$   |
| Minimum reported Hermitian-part eigenvalue | $-6.932262332683731\times10^{-16}$ | At least $-10^{-10}$ |

The [independent oracle](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/independent_qiskit_oracle.py)
constructs scalar gates with Qiskit, starts its density matrix at the physical
all-zero state, and builds standard channels from literal Pauli-Liouvillian
matrix exponentials. Calibration uses Qiskit's average-gate-fidelity function
and fixed bisection. Ballarin lowering, pruning and local Kraus channels are
implemented separately. It imports no YAQS or benchmark numerical helper.
The tests include disabling the selected evaluator's numerical functions and
gate-matrix builders while the independent oracle continues to work.

The [frozen validation protocol](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/experiments/overnight_state_preparation_20260913_0108/INDEPENDENT_CONFIRMATION_VALIDATION_PROTOCOL.md)
requires all methods, targets, replicas, panels and profiles. Expected scores
are used only after independent propagation to calculate discrepancies. The
fixed 600-second CPU admission cap includes preparation and run attempts.
Measured work was **47.715553 CPU seconds** and **68.135319917 wall seconds**,
including preparation of 3.846407 CPU seconds. These are verification costs,
separate from constructing or optimizing the competing methods.

A subsequent read-only audit independently reconstructed every published row
from raw oracle measurements and sealed source endpoints. It found exact
agreement with the published comparison rows, checked physical and Ballarin
gate counts, and recomputed the all-row acceptance result. All **12,080 original
confirmation files** retained both their exact file set and their pre-run byte
hashes. This reduction used only stored scalar diagnostics: it recomputed the
trace defect from its real and imaginary components, but did not perform
another density propagation or eigenvalue calculation. Final density matrices
are not retained in the raw oracle receipts.

The result establishes numerical agreement for these sealed circuits under the
specified benchmark models. It does not establish hardware accuracy, regenerate
the target fixture, change the failed performance criterion, or demonstrate
larger-system scalability.

Evidence is retained in the
[complete verification result](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/overnight-state-preparation-20260913-0108/independent-confirmation-validation/RESULT.json)
and the [independent reduction audit](/Users/maximilianfrohlich/Desktop/GitHub/yaqs/output/overnight-state-preparation-20260913-0108/validation/INDEPENDENT_VERIFICATION_RESULT_AUDIT.json).
The latter's content checksum is
`c0eb7a9f46b123d86b81dd4eb7d265c7edfabee46bd825427c2ac59f7d86a42d`.
The separate read-only reduction consumed 4.255584 CPU seconds; it is not added
to the frozen verification run's measured total.
