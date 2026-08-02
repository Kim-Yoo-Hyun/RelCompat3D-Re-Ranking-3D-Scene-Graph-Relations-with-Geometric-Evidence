# RelCompat3D Result Summary

This summary covers the active `no_family_indicator_v1` method only. Exact
all-K values, paired intervals, controls, and audits are stored in the paths
listed by `manifest.json`.

## Evaluation Contract

- Predictors: VL-SAT, Open3DSG, and SGFN.
- Shared target: 157 3DSSG validation scans, 548 relation contexts, and 3,972
  exact-label ground-truth relations in support/contact, proximity, and
  vertical-order families.
- Metrics: exact-label Recall@K and verifier-derived Violation@K for
  `K={5,10,20,50,100}`.
- Proposed variants: RelCompat3D-Linear and RelCompat3D-MLP under the same
  family-aware re-ranking rule.

## Main K=50 Operating Points

All values are percentages.

| Predictor | Source R/V | Linear R/V | MLP R/V |
| --- | ---: | ---: | ---: |
| VL-SAT | 92.72 / 2.68 | 92.77 / 1.97 | 92.72 / 1.89 |
| Open3DSG | 40.43 / 13.87 | 44.18 / 3.42 | 46.70 / 4.13 |
| SGFN | 74.02 / 3.85 | 74.50 / 2.63 | 74.57 / 2.58 |

Across every reported predictor--K setting, both variants have Recall point
estimates no lower and Violation point estimates no higher than their source
ranking. This is a point-estimate statement; paired scan-level intervals are
reported in the canonical evaluation artifacts.

## Supporting Evidence

- Wrong-predicate, wrong-pair, shuffled-geometry, fixed-predicate swap,
  distance-only, and compatibility-only controls test the method factors for
  both compatibility estimators.
- Matched Linear/MLP component diagnostics show that the linked pairwise term
  has a small, estimator-dependent direct effect. Transformation averaging
  gives zero transformed-view compatibility error and exact transformed
  top-\(K\) membership even when aggregate metrics change little.
- Point- and mesh-based measurements reproduce the direction of the reported
  changes under an alternative geometric construct; they are not an
  independent physical-validity ground truth.
- The compatibility models and split firewall exclude predictor identity,
  predictor score, and final-validation rows from fitting.
- A post-hoc frozen-grid analysis preserves the favorable
  Recall--Violation direction in all 75 Linear and 74/75 MLP conditions over
  five smooth non-identity source-score mappings. A percentile condition
  produces small Recall losses, so this supports bounded robustness rather
  than score-scale invariance.
- At \(K=50\), both learned variants Pareto-dominate the training-positive
  robust-density baseline on all three predictors. Hard-tail and Hard-drop use
  evaluation-verifier labels and remain non-deployable diagnostics. None of
  these analyses selects a new active method.
- The matched `pv_global` route preserves support/contact positions and
  identities but has estimator- and \(K\)-dependent effects relative to active
  family slots. Family slots are therefore interpreted as a
  composition-preserving constraint, not an aggregate-optimal route.
- The construct-dependence package verifies the dependency matrix and the
  linked feature-removal, uncertainty-policy, component-removal, and
  point/mesh evidence. It strengthens the verifier-derived claim but does not
  create independent physical-validity ground truth.
- Five predeclared fitting executions reproduce the Linear model exactly.
  MLP variation is small overall; one VL-SAT \(K=50\) seed loses one
  exact-label relation while still reducing Violation, so the evidence does
  not support seed-uniform Pareto improvement.
- A de-identified row-level reproducer regenerates Tables 1--3 and Figure 3
  data and checks 291 canonical cells with maximum absolute error zero.
- Candidate-pool coverage is 99.72% for VL-SAT and SGFN and 79.68% for
  Open3DSG. At \(K=50\), the active-route oracle reaches 96.73%, 86.05%, and
  63.72% Recall, respectively. These are fixed-candidate diagnostic upper
  bounds, not model results.
- The ReplicaSSG/FROSS result is a previously observed transfer stress test and
  does not establish dataset-level generalization.

## Claim Boundary

The evidence supports a scoped re-ranking result for fixed predictions on one
shared 3DSSG target. It does not claim broad 3D scene graph SOTA,
source-independent score calibration, solved support/contact compatibility, or
independent physical-validity annotation.
