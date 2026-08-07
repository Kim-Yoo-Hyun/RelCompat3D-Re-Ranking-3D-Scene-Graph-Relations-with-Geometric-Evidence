# Family-Aware Routing Constraint Controls

Status: `completed`

The direct matched control is `pv_global`: it uses the same candidates, compatibility estimator, product utility, support/contact slots, and support/contact order as `family_slots`, but merges proximity and vertical-order candidates into one queue.

| Source | Estimator | Route | R@50 | V@50 | R@100 | V@100 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| vlsat | linear | family_slots | 92.77 | 1.97 | 96.58 | 2.95 |
| vlsat | linear | pv_global | 92.72 | 1.95 | 96.60 | 2.94 |
| vlsat | linear | support_order_only | 92.90 | 1.98 | 96.58 | 2.86 |
| vlsat | linear | all_families | 92.93 | 2.03 | 96.88 | 3.25 |
| vlsat | mlp | family_slots | 92.72 | 1.89 | 96.50 | 2.96 |
| vlsat | mlp | pv_global | 92.77 | 1.86 | 96.53 | 2.93 |
| vlsat | mlp | support_order_only | 92.98 | 1.89 | 96.53 | 2.87 |
| vlsat | mlp | all_families | 92.75 | 1.92 | 96.88 | 3.17 |
| open3dsg | linear | family_slots | 44.18 | 3.42 | 56.85 | 3.24 |
| open3dsg | linear | pv_global | 44.76 | 3.39 | 57.83 | 3.08 |
| open3dsg | linear | support_order_only | 43.71 | 4.01 | 54.46 | 3.20 |
| open3dsg | linear | all_families | 46.42 | 2.69 | 60.05 | 3.31 |
| open3dsg | mlp | family_slots | 46.70 | 4.13 | 59.89 | 3.71 |
| open3dsg | mlp | pv_global | 43.78 | 4.16 | 56.09 | 3.68 |
| open3dsg | mlp | support_order_only | 41.52 | 4.60 | 52.95 | 3.74 |
| open3dsg | mlp | all_families | 48.62 | 3.60 | 63.65 | 4.26 |
| sgfn | linear | family_slots | 74.50 | 2.63 | 93.03 | 3.50 |
| sgfn | linear | pv_global | 74.67 | 2.61 | 93.28 | 3.39 |
| sgfn | linear | support_order_only | 74.45 | 2.61 | 92.88 | 3.22 |
| sgfn | linear | all_families | 77.04 | 2.58 | 94.13 | 3.71 |
| sgfn | mlp | family_slots | 74.57 | 2.58 | 92.88 | 3.50 |
| sgfn | mlp | pv_global | 75.25 | 2.57 | 93.13 | 3.37 |
| sgfn | mlp | support_order_only | 74.70 | 2.60 | 92.75 | 3.26 |
| sgfn | mlp | all_families | 77.62 | 2.49 | 94.21 | 3.63 |

`support_order_only` additionally removes fixed support positions while preserving the relative source order of support/contact rows. `all_families` is a scope comparison that also applies compatibility to support/contact and is not a matched test of the family-slot constraint.
