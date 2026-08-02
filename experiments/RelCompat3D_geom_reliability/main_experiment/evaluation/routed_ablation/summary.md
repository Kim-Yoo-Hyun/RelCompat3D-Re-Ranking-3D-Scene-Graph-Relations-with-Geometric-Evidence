# Public/Full Routed Ablation Evaluation

Status: `completed`

All conditions preserve the source family-slot sequence and support/contact order; only proximity and vertical candidates are reordered.

| Source | Condition | R@50 | V@50 | R@100 | V@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| VL-SAT | `source_score` | 0.9272 | 0.0268 | 0.9635 | 0.0476 |
| VL-SAT | `structured_product` | 0.9277 | 0.0197 | 0.9658 | 0.0295 |
| VL-SAT | `wrong_predicate_product` | 0.9046 | 0.0497 | 0.9479 | 0.0795 |
| VL-SAT | `wrong_pair_product` | 0.9056 | 0.0261 | 0.9496 | 0.0473 |
| VL-SAT | `shuffled_geometry_product` | 0.8917 | 0.0305 | 0.9411 | 0.0560 |
| VL-SAT | `endpoint_swap_fixed_label_product` | 0.9046 | 0.0497 | 0.9481 | 0.0795 |
| VL-SAT | `distance_only` | 0.8190 | 0.0534 | 0.8980 | 0.0809 |
| VL-SAT | `compatibility_only` | 0.6765 | 0.0161 | 0.8389 | 0.0203 |
| Open3DSG | `source_score` | 0.4043 | 0.1387 | 0.5111 | 0.1242 |
| Open3DSG | `structured_product` | 0.4418 | 0.0342 | 0.5685 | 0.0324 |
| Open3DSG | `wrong_predicate_product` | 0.4265 | 0.2198 | 0.5347 | 0.2096 |
| Open3DSG | `wrong_pair_product` | 0.3852 | 0.0850 | 0.4932 | 0.0836 |
| Open3DSG | `shuffled_geometry_product` | 0.3842 | 0.1293 | 0.4879 | 0.1242 |
| Open3DSG | `endpoint_swap_fixed_label_product` | 0.4267 | 0.2198 | 0.5363 | 0.2096 |
| Open3DSG | `distance_only` | 0.5116 | 0.0824 | 0.6322 | 0.0955 |
| Open3DSG | `compatibility_only` | 0.4222 | 0.0342 | 0.5687 | 0.0324 |
| SGFN | `source_score` | 0.7402 | 0.0385 | 0.9235 | 0.0630 |
| SGFN | `structured_product` | 0.7450 | 0.0263 | 0.9303 | 0.0350 |
| SGFN | `wrong_predicate_product` | 0.7158 | 0.0942 | 0.8993 | 0.1299 |
| SGFN | `wrong_pair_product` | 0.7155 | 0.0399 | 0.8807 | 0.0665 |
| SGFN | `shuffled_geometry_product` | 0.7059 | 0.0473 | 0.8683 | 0.0836 |
| SGFN | `endpoint_swap_fixed_label_product` | 0.7158 | 0.0942 | 0.9003 | 0.1298 |
| SGFN | `distance_only` | 0.6319 | 0.1000 | 0.8406 | 0.1266 |
| SGFN | `compatibility_only` | 0.5279 | 0.0232 | 0.7064 | 0.0230 |

`compatibility_only` removes the source score only inside the routed proximity/vertical families; support/contact remains a source-order pass-through.
