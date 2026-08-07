# RelCompat3D result summary

## Evaluation setup

- Predictors: VL-SAT, Open3DSG, and SGFN.
- Data: 157 3DSSG validation scans, 548 contexts, and 3,972 exact-match
  ground-truth relations in the evaluated families.
- Metrics: exact-match Recall@K and verifier-derived Violation@K for
  `K={5,10,20,50,100}`.
- Methods: RelCompat3D-Linear and RelCompat3D-MLP under the same family-aware
  re-ranking rule.

## Results at K=50

All values are percentages.

| Predictor | Source R/V | Linear R/V | MLP R/V |
| --- | ---: | ---: | ---: |
| VL-SAT | 92.72 / 2.68 | 92.77 / 1.97 | 92.72 / 1.89 |
| Open3DSG | 40.43 / 13.87 | 44.18 / 3.42 | 46.70 / 4.13 |
| SGFN | 74.02 / 3.85 | 74.50 / 2.63 | 74.57 / 2.58 |

Across the reported predictor--K settings, both variants have Recall point
estimates no lower and Violation point estimates no higher than the source
rankings. Paired scene-level intervals are available in the experiment files.

## Additional evidence

- Wrong-predicate, wrong-pair, shuffled-geometry, fixed-predicate-swap,
  distance-only, and compatibility-only controls test the method inputs.
- Pairwise-loss and transformation-averaging analyses are reported for both
  estimators.
- Point/mesh measurements support the direction of the Violation changes using
  an alternative geometric measurement. They are not independent annotations.
- Predefined source-score transformations, simple baselines, routing controls,
  and five training seeds test the stability of the reported behavior.
- Candidate-pool oracle Recall distinguishes re-ranking limits from missing
  source candidates.
- The table-generation check reproduces 291 reported numerical entries.

## Evaluation scope

The evidence supports re-ranking fixed predictions from three predictors on
one shared 3DSSG validation split. It does not establish broad 3D scene graph
state of the art, dataset-level generalization, or independently annotated
geometric validity.
