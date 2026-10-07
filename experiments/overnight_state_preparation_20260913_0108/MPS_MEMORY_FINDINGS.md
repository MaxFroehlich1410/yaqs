# Retained MPS memory: completed q6 diagnostic

All eight fixed development endpoints passed the separate memory/physics diagnostic. The original parameter-coordinate overlap failure remains unchanged. This engineering check is not a confirmation rerun or a scaling result.

Each measurement reuses the original first-step parameters and exact saved target tensors. The complete original batch map hash was reproduced before taking its predeclared path zero. That path had no noise factors in every case; these are **single-path measurements, not representative full weighted-batch costs**.

| Target                  | Circuit                | Retained tape bytes | Exact target bytes | Tape + target bytes | Retained states | MPS gradient CPU s | Dense gradient CPU s | Max bond |
| ----------------------- | ---------------------- | ------------------: | -----------------: | ------------------: | --------------: | -----------------: | -------------------: | -------: |
| gaussian_mu0p5_sigma0p1 | original108            |              233152 |               2688 |              235840 |             217 |           0.056351 |             0.003919 |        8 |
| gaussian_mu0p5_sigma0p1 | structural_mean2_lbfgs |              140224 |               2688 |              142912 |             187 |           0.040181 |             0.003011 |        4 |
| tfim_critical           | original108            |              259008 |               2688 |              261696 |             217 |           0.049708 |             0.003450 |        8 |
| tfim_critical           | structural_mean2_lbfgs |              136896 |               2688 |              139584 |             187 |           0.041412 |             0.003286 |        4 |
| haar_random_1           | original108            |              336000 |               2688 |              338688 |             217 |           0.048783 |             0.003715 |        8 |
| haar_random_1           | structural_mean2_lbfgs |              313344 |               2688 |              316032 |             193 |           0.042991 |             0.003462 |        8 |
| random_mps_bond2        | original108            |              365952 |               2688 |              368640 |             217 |           0.048481 |             0.003866 |        8 |
| random_mps_bond2        | structural_mean2_lbfgs |              122816 |               2688 |              125504 |             187 |           0.039644 |             0.003026 |        4 |

The eight inclusive endpoint attempts used 1.032438 CPU seconds. Nested gradient costs were 0.367551 seconds for MPS and 0.027735 seconds for dense, with 0.042394 seconds for retained-memory inspection. These nested costs are already included in the inclusive total.

Maximum absolute errors were: gradient 2.637e-15, fidelity/loss 4.663e-15, retained-state norm 5.329e-15, and saved-target reconstruction 1.193e-15; all are below the frozen 1e-10 tolerance.

The helper counts all public forward `states` and `gate_outputs`, deduplicates NumPy backing owners, and adds the exact target only once. No tape/target payload was shared. The earlier 2688-byte number describes all tensors in one MPS state; it is not the full tape. Zero-threshold, uncapped execution reached bond 8. Unreturned adjoints, temporary contractions, gate/noise matrices, Python containers and allocation overhead remain outside the snapshot.

Worker-lifetime peak RSS ranged from 168574976 to 169426944 bytes. It includes saved inputs, dense validation and earlier endpoints; it cannot be interpreted as incremental or peak MPS algorithm memory.

This small-system result establishes measurement scope and confirms overlapping backend physics. It does not establish memory superiority over a dense tape, and it does not justify q12 optimization after the candidate failed its q6 confirmation gate.

Protocol content checksum: `7fd51cc6585521fc1d5fcffbb03fa9b17cf1e29b793fc8f78b00d94d777a359a`. Raw measurements, gradients, full timing receipts and completion seals remain in each endpoint directory. See `ANALYSIS.json` for machine-readable reductions and `../validation/MPS_MEMORY_AUTHENTICATION.json` for the read-only audit.
