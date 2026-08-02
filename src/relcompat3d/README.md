# RelCompat3D Python Modules

The filenames use a short `verb_target.py` convention. Docker services call the
training, evaluation, audit, and artifact entry points directly; shared modules
provide features, relation-consistency logic, controls, metrics, and paths. All
modules use file-relative imports and the packages pinned in
`configs/relcompat3d/Dockerfile`.

## Functional Groups

- Training: `build_training_rows.py`, `fit_train_only.py`, `fit_linear.py`,
  `fit_mlp.py`, and `fit_factor_controls.py`.
- Core logic: `compatibility_features.py`, `relation_consistency.py`,
  `control_utils.py`, `evaluate_metrics.py`, and `paths.py`.
- Main evaluation: `evaluate_main.py`, `evaluate_comparators.py`,
  `evaluate_linear_controls.py`, `evaluate_mlp_controls.py`, and
  `evaluate_component_removals.py`.
- Family and uncertainty checks: `evaluate_support_order.py`,
  `evaluate_support_intervals.py`, and `evaluate_scan_intervals.py`.
- Construct checks: `audit_point_mesh.py`, `audit_mlp_point_mesh.py`,
  `evaluate_feature_removal.py`, `evaluate_counterfactuals.py`, and
  `build_construct_package.py`.
- Additional evidence: `evaluate_score_robustness.py`,
  `evaluate_routing_constraints.py`, `evaluate_components.py`,
  `evaluate_seeds.py`, `benchmark_runtime.py`, `evaluate_open3dsg.py`, and
  `evaluate_transfer.py`. Shared training-control helpers are in
  `training_control_utils.py`.
- Row-level verification: `build_reproduction_rows.py`,
  `reproduce_from_rows.py`, and `evaluate_candidate_oracle.py`.

Generated rows, caches, checkpoints, and model payloads do not belong under
`src/`.
