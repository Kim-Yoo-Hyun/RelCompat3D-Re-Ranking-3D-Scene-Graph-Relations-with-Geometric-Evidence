# Same-Route Fusion Comparator Evaluation

Status: `completed`

All methods use the same family-slot composition, support/contact pass-through, official 548-context universe, and scan-cluster resampling.

| Source | Method | R@50 | V@50 | R@100 | V@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| vlsat | source_score | 0.9272 | 0.0268 | 0.9635 | 0.0476 |
| vlsat | routed_product | 0.9277 | 0.0197 | 0.9658 | 0.0295 |
| vlsat | routed_rank_average | 0.9023 | 0.0162 | 0.9705 | 0.0226 |
| vlsat | routed_rrf | 0.9172 | 0.0182 | 0.9705 | 0.0250 |
| vlsat | routed_matched_mlp | 0.9272 | 0.0189 | 0.9650 | 0.0296 |
| open3dsg | source_score | 0.4043 | 0.1387 | 0.5111 | 0.1242 |
| open3dsg | routed_product | 0.4418 | 0.0342 | 0.5685 | 0.0324 |
| open3dsg | routed_rank_average | 0.4320 | 0.0460 | 0.5400 | 0.0549 |
| open3dsg | routed_rrf | 0.4245 | 0.0960 | 0.5468 | 0.0780 |
| open3dsg | routed_matched_mlp | 0.4670 | 0.0413 | 0.5989 | 0.0371 |
| sgfn | source_score | 0.7402 | 0.0385 | 0.9235 | 0.0630 |
| sgfn | routed_product | 0.7450 | 0.0263 | 0.9303 | 0.0350 |
| sgfn | routed_rank_average | 0.7022 | 0.0235 | 0.9272 | 0.0260 |
| sgfn | routed_rrf | 0.6850 | 0.0254 | 0.8799 | 0.0307 |
| sgfn | routed_matched_mlp | 0.7457 | 0.0258 | 0.9288 | 0.0350 |
