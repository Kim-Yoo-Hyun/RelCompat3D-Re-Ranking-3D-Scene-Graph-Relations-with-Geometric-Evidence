# RelCompat3D Component Diagnostics

Status: `completed`

Full, no-pairwise-loss, and no-transformation-averaging are matched within each estimator. All results use the fixed candidates and family-aware route.

## Held-out Linked-Pair Diagnostics

| Estimator | Condition | Pairs | Positive wins | Mean margin | P05 | Median | P95 | Softplus loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | full | 3516 | 0.992605 | 14.141347 | 4.139685 | 13.346198 | 27.898860 | 0.035343 |
| linear | no_pairwise | 3516 | 0.991752 | 13.614517 | 4.016987 | 12.735990 | 26.952750 | 0.036074 |
| mlp | full | 3516 | 0.993174 | 12.149095 | 3.931722 | 11.621409 | 21.863128 | 0.037985 |
| mlp | no_pairwise | 3516 | 0.992605 | 12.648925 | 3.740040 | 12.032562 | 22.540572 | 0.037813 |

## Aggregate Point Estimates

| Predictor | Condition | R@50 | V@50 | R@100 | V@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| vlsat | linear_full | 0.9277 | 0.0197 | 0.9658 | 0.0295 |
| vlsat | linear_no_pairwise | 0.9277 | 0.0198 | 0.9658 | 0.0297 |
| vlsat | linear_no_averaging | 0.9277 | 0.0197 | 0.9660 | 0.0295 |
| vlsat | mlp_full | 0.9272 | 0.0189 | 0.9650 | 0.0296 |
| vlsat | mlp_no_pairwise | 0.9275 | 0.0180 | 0.9648 | 0.0280 |
| vlsat | mlp_no_averaging | 0.9275 | 0.0189 | 0.9645 | 0.0296 |
| open3dsg | linear_full | 0.4418 | 0.0342 | 0.5685 | 0.0324 |
| open3dsg | linear_no_pairwise | 0.4418 | 0.0342 | 0.5692 | 0.0324 |
| open3dsg | linear_no_averaging | 0.4413 | 0.0342 | 0.5692 | 0.0324 |
| open3dsg | mlp_full | 0.4670 | 0.0413 | 0.5989 | 0.0371 |
| open3dsg | mlp_no_pairwise | 0.4600 | 0.0399 | 0.5939 | 0.0362 |
| open3dsg | mlp_no_averaging | 0.4507 | 0.0398 | 0.5735 | 0.0368 |
| sgfn | linear_full | 0.7450 | 0.0263 | 0.9303 | 0.0350 |
| sgfn | linear_no_pairwise | 0.7450 | 0.0264 | 0.9303 | 0.0353 |
| sgfn | linear_no_averaging | 0.7450 | 0.0263 | 0.9303 | 0.0350 |
| sgfn | mlp_full | 0.7457 | 0.0258 | 0.9288 | 0.0350 |
| sgfn | mlp_no_pairwise | 0.7452 | 0.0255 | 0.9295 | 0.0330 |
| sgfn | mlp_no_averaging | 0.7457 | 0.0258 | 0.9300 | 0.0352 |

Transformation-error distributions and transformed-view top-K membership checks are recorded in `summary.json` and the CSV files.

