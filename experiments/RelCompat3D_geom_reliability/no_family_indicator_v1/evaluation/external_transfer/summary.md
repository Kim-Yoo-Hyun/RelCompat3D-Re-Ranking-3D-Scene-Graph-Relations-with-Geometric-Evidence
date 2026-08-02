# ReplicaSSG Final-Method Cross-Dataset Evaluation

Status: `blocked_external_dataset_evaluation`

This is a benchmark evaluation of the locked final method on a previously observed external target; it is not untouched prospective confirmation.

| Method | R@10 | V@10 | R@50 | V@50 | R@100 | V@100 | dR@100 | dV@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source_score | 0.07558 | 0.08182 | 0.26163 | 0.13284 | 0.35465 | 0.19674 | +0.00000 | +0.00000 |
| routed_product | 0.14535 | 0.02727 | 0.31395 | 0.09041 | 0.35465 | 0.19578 | +0.00000 | -0.00096 |
| unrestricted_product | 0.13953 | 0.02727 | 0.31395 | 0.09410 | 0.35465 | 0.19578 | +0.00000 | -0.00096 |
| routed_rank_average | 0.13372 | 0.00000 | 0.34302 | 0.02399 | 0.38372 | 0.04223 | +0.02907 | -0.15451 |
| global_rank_average | 0.09302 | 0.00000 | 0.23256 | 0.02030 | 0.31977 | 0.04223 | -0.03488 | -0.15451 |
| global_rrf_c60 | 0.10465 | 0.00000 | 0.22093 | 0.04613 | 0.32558 | 0.06142 | -0.02907 | -0.13532 |
| routed_compatibility_only | 0.09302 | 0.00000 | 0.27907 | 0.01476 | 0.37209 | 0.02399 | +0.01744 | -0.17274 |

Primary joint gate: `fail`.

## Negative-transfer decomposition

- Source-score zeros: 86.280%; ones: 10.808%; distinct values: 105.
- External feature cells with |train-standardized z|>3: 19.200%; missing: 3.509%.
- Exact-GT AUC: source=0.7731, compatibility=0.6686, product=0.7982.
- Verifier-satisfaction AUC: source=0.5720, compatibility=0.9460, product=0.5758.
- Scope: 172 exact-label GT relations, three predicates, two families, and 11 scene-level bootstrap units.

Full feature-shift, rank-displacement, per-scene, family, and selection-transition diagnostics are in `summary.json`.
