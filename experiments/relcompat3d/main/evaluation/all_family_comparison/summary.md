# All-Family Comparison

Status: `completed`

The main paper compatibility is relation-algebra-constrained compatibility; `orbit_pairwise_projected` is retained only as its artifact ID.

## K=100 overall

| Source | Method | Recall | verifier V | uncertainty | pessimistic V |
| --- | --- | ---: | ---: | ---: | ---: |
| VL-SAT | source_score | 0.9635 | 0.0476 | 0.3464 | 0.3941 |
| VL-SAT | all_family_product | 0.9688 | 0.0325 | 0.3131 | 0.3455 |
| VL-SAT | rank_average_all_families | 0.9610 | 0.0248 | 0.2682 | 0.2929 |
| VL-SAT | rrf_all_families | 0.9602 | 0.0232 | 0.2500 | 0.2732 |
| VL-SAT | pooled_product | 0.9690 | 0.0387 | 0.3133 | 0.3520 |
| VL-SAT | hard_rule_filter | 0.9627 | 0.0000 | 0.3712 | 0.3712 |
| VL-SAT | family_product_continuity | 0.9688 | 0.0328 | 0.3138 | 0.3466 |
| VL-SAT | compatibility_only_all_families | 0.6130 | 0.0198 | 0.1956 | 0.2154 |
| Open3DSG | source_score | 0.5161 | 0.1242 | 0.4164 | 0.5406 |
| Open3DSG | all_family_product | 0.6052 | 0.0340 | 0.2724 | 0.3063 |
| Open3DSG | rank_average_all_families | 0.5987 | 0.0527 | 0.2932 | 0.3459 |
| Open3DSG | rrf_all_families | 0.5954 | 0.0788 | 0.2968 | 0.3757 |
| Open3DSG | pooled_product | 0.6443 | 0.0747 | 0.2703 | 0.3450 |
| Open3DSG | hard_rule_filter | 0.5368 | 0.0000 | 0.4722 | 0.4722 |
| Open3DSG | family_product_continuity | 0.6050 | 0.0339 | 0.2707 | 0.3046 |
| Open3DSG | compatibility_only_all_families | 0.5725 | 0.0322 | 0.2332 | 0.2654 |
| SGFN | source_score | 0.9235 | 0.0630 | 0.3732 | 0.4362 |
| SGFN | all_family_product | 0.9413 | 0.0371 | 0.3399 | 0.3770 |
| SGFN | rank_average_all_families | 0.9459 | 0.0269 | 0.2839 | 0.3108 |
| SGFN | rrf_all_families | 0.9063 | 0.0267 | 0.2788 | 0.3054 |
| SGFN | pooled_product | 0.9413 | 0.0464 | 0.3357 | 0.3821 |
| SGFN | hard_rule_filter | 0.9270 | 0.0000 | 0.4042 | 0.4042 |
| SGFN | family_product_continuity | 0.9416 | 0.0376 | 0.3404 | 0.3780 |
| SGFN | compatibility_only_all_families | 0.6130 | 0.0198 | 0.1956 | 0.2154 |

All scores use the same 548 contexts and 3,972 exact-label denominator. The hard-rule diagnostic retains satisfied and uncertain point-subtype rows, adds no synthetic rows, and therefore may select fewer than K candidates.
