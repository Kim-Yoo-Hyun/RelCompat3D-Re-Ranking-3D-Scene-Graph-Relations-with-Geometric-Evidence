# Score Robustness and Simple Baselines

Status: `completed`

This post-hoc analysis uses the exact active candidate pool, source rows, family-slot route, model locks, and scan-cluster bootstrap protocol.

## Canonical rerun gate

- Identity Linear/MLP and Source match the active routed-comparator points: `True`.
- Archived Tier-B hashes match the active manifests: `True`.

## K=50 operating points

| Predictor | Method | Recall | Violation | Decidable V | Uncertainty | Selected |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| vlsat | source | 0.9272 | 0.0268 | 0.0368 | 0.2740 | 27400 |
| vlsat | linear__identity | 0.9277 | 0.0197 | 0.0272 | 0.2752 | 27400 |
| vlsat | mlp__identity | 0.9272 | 0.0189 | 0.0260 | 0.2732 | 27400 |
| vlsat | hard_tail | 0.9270 | 0.0161 | 0.0223 | 0.2784 | 27400 |
| vlsat | hard_drop | 0.9257 | 0.0000 | 0.0000 | 0.2850 | 27400 |
| vlsat | positive_density | 0.9177 | 0.0268 | 0.0371 | 0.2785 | 27400 |
| open3dsg | source | 0.4043 | 0.1387 | 0.2271 | 0.3892 | 26636 |
| open3dsg | linear__identity | 0.4418 | 0.0342 | 0.0540 | 0.3662 | 26636 |
| open3dsg | mlp__identity | 0.4670 | 0.0413 | 0.0675 | 0.3879 | 26636 |
| open3dsg | hard_tail | 0.4136 | 0.0342 | 0.0593 | 0.4231 | 26636 |
| open3dsg | hard_drop | 0.4242 | 0.0000 | 0.0000 | 0.4636 | 26631 |
| open3dsg | positive_density | 0.4376 | 0.0509 | 0.0862 | 0.4095 | 26636 |
| sgfn | source | 0.7402 | 0.0385 | 0.0606 | 0.3655 | 27400 |
| sgfn | linear__identity | 0.7450 | 0.0263 | 0.0415 | 0.3665 | 27400 |
| sgfn | mlp__identity | 0.7457 | 0.0258 | 0.0404 | 0.3624 | 27400 |
| sgfn | hard_tail | 0.7404 | 0.0232 | 0.0370 | 0.3715 | 27400 |
| sgfn | hard_drop | 0.7462 | 0.0000 | 0.0000 | 0.3807 | 27400 |
| sgfn | positive_density | 0.7387 | 0.0401 | 0.0645 | 0.3779 | 27400 |

## Interpretation guardrails

- Monotonic mappings are fixed sensitivity conditions, not tuned alternatives.
- Hard-tail and Hard-drop directly consume evaluation-verifier labels and are upper diagnostics, not deployable baselines.
- Positive-density is fitted only from training-split positive geometry and is the closest non-learned continuous baseline.
- A result may support robustness over the tested mappings, but never score-scale invariance.
