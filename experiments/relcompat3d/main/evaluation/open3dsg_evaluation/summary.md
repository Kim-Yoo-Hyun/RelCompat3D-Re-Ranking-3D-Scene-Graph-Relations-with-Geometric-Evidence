# Open3DSG Public-Route Sensitivity

Status: `completed`

| Route | Contexts | GT | Method | R@100 | V@100 |
| --- | ---: | ---: | --- | ---: | ---: |
| official_eligible_533 | 533 | 3899 | source_score | 0.5206 | 0.1242 |
| official_eligible_533 | 533 | 3899 | family_slot_rerank | 0.5791 | 0.0324 |
| official_eligible_533 | 533 | 3899 | all_family_product | 0.6117 | 0.0331 |
| official_eligible_533 | 533 | 3899 | rank_average_all_families | 0.6040 | 0.0522 |
| official_eligible_533 | 533 | 3899 | rrf_all_families | 0.6009 | 0.0784 |
| official_eligible_533 | 533 | 3899 | pooled_product | 0.6522 | 0.0743 |
| official_eligible_533 | 533 | 3899 | hard_rule_filter | 0.5419 | 0.0000 |
| official_full_548 | 548 | 3972 | source_score | 0.5111 | 0.1242 |
| official_full_548 | 548 | 3972 | family_slot_rerank | 0.5685 | 0.0324 |
| official_full_548 | 548 | 3972 | all_family_product | 0.6005 | 0.0331 |
| official_full_548 | 548 | 3972 | rank_average_all_families | 0.5929 | 0.0522 |
| official_full_548 | 548 | 3972 | rrf_all_families | 0.5899 | 0.0784 |
| official_full_548 | 548 | 3972 | pooled_product | 0.6402 | 0.0743 |
| official_full_548 | 548 | 3972 | hard_rule_filter | 0.5320 | 0.0000 |
| recovered_full_548 | 548 | 3972 | source_score | 0.5161 | 0.1242 |
| recovered_full_548 | 548 | 3972 | family_slot_rerank | 0.5735 | 0.0332 |
| recovered_full_548 | 548 | 3972 | all_family_product | 0.6052 | 0.0340 |
| recovered_full_548 | 548 | 3972 | rank_average_all_families | 0.5987 | 0.0527 |
| recovered_full_548 | 548 | 3972 | rrf_all_families | 0.5954 | 0.0788 |
| recovered_full_548 | 548 | 3972 | pooled_product | 0.6443 | 0.0747 |
| recovered_full_548 | 548 | 3972 | hard_rule_filter | 0.5368 | 0.0000 |

The public route drops 15 contexts because fewer than four annotated objects retain view metadata. The 533-context row is the faithful unmodified-pipeline evaluation; the strict-548 row is the conservative common-target sensitivity; the recovered row is reported separately.
