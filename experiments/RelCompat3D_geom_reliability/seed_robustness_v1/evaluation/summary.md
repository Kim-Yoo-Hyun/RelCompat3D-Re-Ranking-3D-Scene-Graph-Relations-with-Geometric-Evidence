# RelCompat3D Five-Seed Robustness

Status: `completed`

The model seeds were fixed before evaluation. The constructed training rows and their counterfactual links are held fixed. The active MLP seed is included but was not reselected from this analysis.

| Estimator | Predictor | K | Recall mean±std | Violation mean±std | Favorable seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| linear | vlsat | 5 | 0.420695 ± 0.000000 | 0.001460 ± 0.000000 | 5/5 |
| linear | vlsat | 10 | 0.633938 ± 0.000000 | 0.005657 ± 0.000000 | 5/5 |
| linear | vlsat | 20 | 0.808157 ± 0.000000 | 0.011405 ± 0.000000 | 5/5 |
| linear | vlsat | 50 | 0.927744 ± 0.000000 | 0.019745 ± 0.000000 | 5/5 |
| linear | vlsat | 100 | 0.965760 ± 0.000000 | 0.029489 ± 0.000000 | 5/5 |
| linear | open3dsg | 5 | 0.037261 ± 0.000000 | 0.009381 ± 0.000000 | 5/5 |
| linear | open3dsg | 10 | 0.113797 ± 0.000000 | 0.023265 ± 0.000000 | 5/5 |
| linear | open3dsg | 20 | 0.236153 ± 0.000000 | 0.031332 ± 0.000000 | 5/5 |
| linear | open3dsg | 50 | 0.441843 ± 0.000000 | 0.034239 ± 0.000000 | 5/5 |
| linear | open3dsg | 100 | 0.568479 ± 0.000000 | 0.032392 ± 0.000000 | 5/5 |
| linear | sgfn | 5 | 0.311682 ± 0.000000 | 0.023723 ± 0.000000 | 5/5 |
| linear | sgfn | 10 | 0.397533 ± 0.000000 | 0.034672 ± 0.000000 | 5/5 |
| linear | sgfn | 20 | 0.491440 ± 0.000000 | 0.029745 ± 0.000000 | 5/5 |
| linear | sgfn | 50 | 0.744965 ± 0.000000 | 0.026314 ± 0.000000 | 5/5 |
| linear | sgfn | 100 | 0.930262 ± 0.000000 | 0.035018 ± 0.000000 | 5/5 |
| mlp | vlsat | 5 | 0.420796 ± 0.000409 | 0.001460 ± 0.000000 | 5/5 |
| mlp | vlsat | 10 | 0.634340 ± 0.000467 | 0.005182 ± 0.000146 | 5/5 |
| mlp | vlsat | 20 | 0.808912 ± 0.000159 | 0.011022 ± 0.000477 | 5/5 |
| mlp | vlsat | 50 | 0.927241 ± 0.000159 | 0.018664 ± 0.001113 | 4/5 |
| mlp | vlsat | 100 | 0.965206 ± 0.000101 | 0.029215 ± 0.002688 | 5/5 |
| mlp | open3dsg | 5 | 0.037059 ± 0.000247 | 0.022664 ± 0.013778 | 5/5 |
| mlp | open3dsg | 10 | 0.117069 ± 0.000747 | 0.032795 ± 0.008737 | 5/5 |
| mlp | open3dsg | 20 | 0.241843 ± 0.003238 | 0.036529 ± 0.004712 | 5/5 |
| mlp | open3dsg | 50 | 0.458610 ± 0.004233 | 0.037153 ± 0.002267 | 5/5 |
| mlp | open3dsg | 100 | 0.592951 ± 0.004502 | 0.034954 ± 0.001330 | 5/5 |
| mlp | sgfn | 5 | 0.311682 ± 0.000000 | 0.023723 ± 0.000000 | 5/5 |
| mlp | sgfn | 10 | 0.397533 ± 0.000000 | 0.034672 ± 0.000000 | 5/5 |
| mlp | sgfn | 20 | 0.491944 ± 0.000159 | 0.029489 ± 0.000036 | 5/5 |
| mlp | sgfn | 50 | 0.746022 ± 0.000294 | 0.025664 ± 0.000453 | 5/5 |
| mlp | sgfn | 100 | 0.929607 ± 0.000565 | 0.034372 ± 0.003012 | 5/5 |

Linear uses deterministic zero initialization and full-batch optimization, so the five declared seed labels reproduce one model hash. MLP varies only its initialization seed.

