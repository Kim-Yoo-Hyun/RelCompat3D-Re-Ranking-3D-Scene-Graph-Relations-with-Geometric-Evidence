# RelCompat3D-MLP Ablation Evaluation

Status: `completed`

All conditions use the fitted MLP compatibility estimator on the 548 evaluation contexts with the same family-aware ranking rule. Support/contact candidates remain in source order.

| Predictor | Condition | R@50 | V@50 | R@100 | V@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| VL-SAT | `source_score` | 0.9272 | 0.0268 | 0.9635 | 0.0476 |
| VL-SAT | `relcompat3d_mlp` | 0.9272 | 0.0189 | 0.9650 | 0.0296 |
| VL-SAT | `mlp_wrong_predicate` | 0.9099 | 0.0503 | 0.9499 | 0.0831 |
| VL-SAT | `mlp_wrong_pair` | 0.9131 | 0.0262 | 0.9562 | 0.0471 |
| VL-SAT | `mlp_shuffled_geometry` | 0.9026 | 0.0300 | 0.9471 | 0.0549 |
| VL-SAT | `mlp_fixed_label_endpoint_swap` | 0.9099 | 0.0499 | 0.9509 | 0.0813 |
| VL-SAT | `distance_only` | 0.8190 | 0.0534 | 0.8980 | 0.0809 |
| VL-SAT | `mlp_compatibility_only` | 0.7784 | 0.0161 | 0.9018 | 0.0205 |
| Open3DSG | `source_score` | 0.4043 | 0.1387 | 0.5111 | 0.1242 |
| Open3DSG | `relcompat3d_mlp` | 0.4670 | 0.0413 | 0.5989 | 0.0371 |
| Open3DSG | `mlp_wrong_predicate` | 0.4678 | 0.1930 | 0.5889 | 0.1834 |
| Open3DSG | `mlp_wrong_pair` | 0.3802 | 0.0931 | 0.4990 | 0.0890 |
| Open3DSG | `mlp_shuffled_geometry` | 0.3683 | 0.1288 | 0.4794 | 0.1238 |
| Open3DSG | `mlp_fixed_label_endpoint_swap` | 0.4552 | 0.1913 | 0.5743 | 0.1815 |
| Open3DSG | `distance_only` | 0.5116 | 0.0824 | 0.6322 | 0.0955 |
| Open3DSG | `mlp_compatibility_only` | 0.4819 | 0.0342 | 0.6133 | 0.0328 |
| SGFN | `source_score` | 0.7402 | 0.0385 | 0.9235 | 0.0630 |
| SGFN | `relcompat3d_mlp` | 0.7457 | 0.0258 | 0.9288 | 0.0350 |
| SGFN | `mlp_wrong_predicate` | 0.7178 | 0.0967 | 0.9001 | 0.1352 |
| SGFN | `mlp_wrong_pair` | 0.7210 | 0.0391 | 0.8905 | 0.0659 |
| SGFN | `mlp_shuffled_geometry` | 0.7115 | 0.0459 | 0.8784 | 0.0806 |
| SGFN | `mlp_fixed_label_endpoint_swap` | 0.7185 | 0.0965 | 0.9023 | 0.1335 |
| SGFN | `distance_only` | 0.6319 | 0.1000 | 0.8406 | 0.1266 |
| SGFN | `mlp_compatibility_only` | 0.5710 | 0.0233 | 0.8099 | 0.0231 |

`distance_only` is head-independent and is retained once as a common control. `mlp_compatibility_only` removes the predictor score only from proximity/vertical ordering; it is not a raw-geometry-only model.
