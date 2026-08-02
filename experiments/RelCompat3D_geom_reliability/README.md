# RelCompat3D experiments

This directory contains the active method and the analyses reported in the
paper and supplement.

| Directory | Purpose |
| --- | --- |
| `no_family_indicator_v1/` | active fitting and main evaluation protocols |
| `row_reproduction_v1/` | Tables 1--3 and Figure 3 regeneration |
| `score_robustness_v1/` | source-score mappings and simple baselines |
| `routing_controls_v1/` | family-aware routing controls |
| `construct_dependence_v1/` | measurement-dependence evidence index |
| `component_diagnostics_v1/` | pairwise-loss and transformation checks |
| `seed_robustness_v1/` | five-seed fitting analysis |
| `candidate_oracle_v1/` | fixed-candidate Recall upper bounds |
| `factor_isolation_protocol/` | factor-separation protocol and locks |
| `train_only_reestablishment_v1/` | split firewall and train-only provenance |

`evaluation/` directories are frozen references. Reproduction services write
to ignored `regenerated/` directories. Fitted JSON models are restored with
`scripts/download_models.sh`.

The active evaluation uses 157 validation scans, 548 contexts, and 3,972
exact-match ground-truth relations. The model archive, licensed inputs, and
commands are documented in `docs/`.
