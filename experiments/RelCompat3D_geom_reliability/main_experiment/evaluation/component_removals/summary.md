# Direct Component Removals

Status: `completed`

All conditions use RelCompat3D-Linear, the same frozen train rows and family-slot route.

| Source | Condition | R@50 | V@50 | R@100 | V@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| vlsat | source_score | 0.9272 | 0.0268 | 0.9635 | 0.0476 |
| vlsat | full_linear | 0.9277 | 0.0197 | 0.9658 | 0.0295 |
| vlsat | no_pairwise_loss | 0.9277 | 0.0198 | 0.9658 | 0.0297 |
| vlsat | no_transformation_averaging | 0.9277 | 0.0197 | 0.9660 | 0.0295 |
| open3dsg | source_score | 0.4043 | 0.1387 | 0.5111 | 0.1242 |
| open3dsg | full_linear | 0.4418 | 0.0342 | 0.5685 | 0.0324 |
| open3dsg | no_pairwise_loss | 0.4418 | 0.0342 | 0.5692 | 0.0324 |
| open3dsg | no_transformation_averaging | 0.4413 | 0.0342 | 0.5692 | 0.0324 |
| sgfn | source_score | 0.7402 | 0.0385 | 0.9235 | 0.0630 |
| sgfn | full_linear | 0.7450 | 0.0263 | 0.9303 | 0.0350 |
| sgfn | no_pairwise_loss | 0.7450 | 0.0264 | 0.9303 | 0.0353 |
| sgfn | no_transformation_averaging | 0.7450 | 0.0263 | 0.9303 | 0.0350 |
