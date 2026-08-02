# Result artifacts

The repository tracks compact outputs used by the paper and supplement. Raw
prediction rows, geometry, and checkpoints remain external.

## Main artifacts

| Artifact | Location |
| --- | --- |
| Tables 1--3 | `experiments/RelCompat3D_geom_reliability/row_reproduction_v1/evaluation/` |
| Figure 3 data and rendering | same directory |
| Active method lock | `experiments/RelCompat3D_geom_reliability/active_method.json` |
| Main model/evaluation protocols | `experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/` |
| Paper-facing evidence map | `results/relcompat3d_geom_reliability/manifest.json` |
| Compact claim summary | `results/relcompat3d_geom_reliability/report.md` |

## Extended evidence

Versioned directories contain score-mapping sensitivity, simple baselines,
routing controls, component removals, training-seed checks, construct
dependence, point/mesh audits, and candidate-pool oracle Recall. These analyses
do not change the active method lock.

## Evaluation scope

The reported comparison uses fixed candidates from VL-SAT, Open3DSG, and SGFN
on the shared 3DSSG validation scenes. Metrics are exact-match Recall at
`K in {5, 10, 20, 50, 100}` and verifier-derived Violation at the same cutoffs.
The evidence supports a scoped relation-reliability layer. It is not a claim of
broad 3D scene graph state of the art, dataset-level generalization, or
independent physical-validity ground truth.

## Frozen versus regenerated outputs

`evaluation/` directories contain immutable reference outputs. New runs write
to `regenerated/`, which is ignored by Git. This prevents a local test from
silently overwriting the evidence used by the paper.
