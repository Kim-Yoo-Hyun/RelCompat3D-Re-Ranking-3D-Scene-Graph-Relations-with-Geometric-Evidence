# Results

The `relcompat3d_geom_reliability/` directory provides a small index over the
reported experiment outputs. Numerical artifacts remain in their versioned
experiment directories to avoid duplicate copies.

## Main artifacts

| Artifact | Location |
| --- | --- |
| Tables 1--3 | `../experiments/RelCompat3D_geom_reliability/paper_reproduction/evaluation/` |
| Figure 3 data and rendering | `../experiments/RelCompat3D_geom_reliability/paper_reproduction/evaluation/` |
| Active method lock | `../experiments/RelCompat3D_geom_reliability/active_method.json` |
| Main protocols and evaluation | `../experiments/RelCompat3D_geom_reliability/main_experiment/` |
| Paper-facing evidence map | `relcompat3d_geom_reliability/manifest.json` |
| Compact claim summary | `relcompat3d_geom_reliability/report.md` |

The experiment directories also contain score-mapping sensitivity, simple
baselines, routing controls, component removals, training-seed checks,
measurement-dependence checks, point/mesh audits, and candidate-pool oracle
Recall. These analyses do not change the active method lock.

## Evaluation scope

The reported comparison uses fixed candidates from VL-SAT, Open3DSG, and SGFN
on the shared 3DSSG validation scenes. It reports exact-match Recall and
verifier-derived Violation at `K in {5, 10, 20, 50, 100}`. The evidence supports
a scoped relation-reliability layer, not broad 3D scene graph state of the art,
dataset-level generalization, or independently annotated geometry validity.

Reference outputs remain in `evaluation/`. New runs write to ignored
`regenerated/` directories so local tests do not overwrite the evidence used
by the paper.
