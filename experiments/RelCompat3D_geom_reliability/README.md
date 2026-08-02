# RelCompat3D experiments

This directory contains the active method and the analyses reported in the
paper and supplement.

| Directory | Purpose |
| --- | --- |
| `main_experiment/` | active fitting and main evaluation protocols |
| `paper_reproduction/` | Tables 1--3 and Figure 3 regeneration |
| `score_robustness/` | source-score mappings and simple baselines |
| `routing_controls/` | family-aware routing controls |
| `measurement_audit/` | measurement-dependence evidence index |
| `component_analysis/` | pairwise-loss and transformation checks |
| `seed_robustness/` | five-seed fitting analysis |
| `candidate_oracle/` | fixed-candidate Recall upper bounds |
| `factor_controls/` | factor-separation protocol and locks |
| `training_protocol/` | split firewall and train-only provenance |

`evaluation/` directories are frozen references. Reproduction services write
to ignored `regenerated/` directories. Fitted JSON models are restored with
`scripts/download_models.sh`.

The active evaluation uses 157 validation scans, 548 contexts, and 3,972
exact-match ground-truth relations. The model archive, licensed inputs, and
commands are documented in `docs/`.
