# MLP Comparison

Status: `completed`

| Source | Method | Supervision | R@100 | V@100 |
| --- | --- | --- | ---: | ---: |
| vlsat | all_family_product | shared compatibility target | 0.9688 | 0.0325 |
| vlsat | shared_mlp_bce_product | shared compatibility target | 0.9688 | 0.0314 |
| vlsat | shared_mlp_pairwise_product | shared compatibility target | 0.9688 | 0.0317 |
| vlsat | source-specific nonlinear | SGFN exact-label correctness | 0.9625 | 0.0311 |
| open3dsg | all_family_product | shared compatibility target | 0.6052 | 0.0340 |
| open3dsg | shared_mlp_bce_product | shared compatibility target | 0.6276 | 0.0430 |
| open3dsg | shared_mlp_pairwise_product | shared compatibility target | 0.6415 | 0.0434 |
| open3dsg | source-specific nonlinear | SGFN exact-label correctness | 0.6166 | 0.0334 |
| sgfn | all_family_product | shared compatibility target | 0.9413 | 0.0371 |
| sgfn | shared_mlp_bce_product | shared compatibility target | 0.9421 | 0.0356 |
| sgfn | shared_mlp_pairwise_product | shared compatibility target | 0.9421 | 0.0363 |
| sgfn | source-specific nonlinear | SGFN exact-label correctness | 0.9466 | 0.0279 |

## Paired K=100 comparison with RelCompat3D-Linear

| Source | MLP method | delta Recall (95% CI) | delta V (95% CI) |
| --- | --- | ---: | ---: |
| vlsat | shared_mlp_bce_product | +0.0000 [-0.0027, +0.0019] | -0.0011 [-0.0023, +0.0001] |
| vlsat | shared_mlp_pairwise_product | +0.0000 [-0.0027, +0.0019] | -0.0008 [-0.0018, +0.0004] |
| open3dsg | shared_mlp_bce_product | +0.0224 [+0.0140, +0.0313] | +0.0091 [+0.0071, +0.0109] |
| open3dsg | shared_mlp_pairwise_product | +0.0363 [+0.0252, +0.0482] | +0.0095 [+0.0078, +0.0112] |
| sgfn | shared_mlp_bce_product | +0.0008 [-0.0016, +0.0031] | -0.0015 [-0.0027, -0.0002] |
| sgfn | shared_mlp_pairwise_product | +0.0008 [-0.0013, +0.0030] | -0.0009 [-0.0020, +0.0003] |

The shared MLP compatibility models use no source score or predictor identity and are applied unchanged to all three predictors. The exact-label rescorer is reported separately because it uses SGFN-specific correctness supervision.
