# RelCompat3D Python Modules

The filenames use a short `verb_target.py` convention. Docker services call the
training, evaluation, audit, and artifact entry points directly; shared modules
provide features, relation-consistency logic, controls, metrics, and paths. All
modules use file-relative imports and the packages pinned in
`configs/relcompat3d/Dockerfile`.

## Functional Groups

- Training: `build_training_rows.py`, `fit_train_only.py`, `fit_linear.py`,
  `fit_mlp.py`, and `fit_factor_controls.py`.
- Source preprocessing: `adapt_source_predictions.py` converts VL-SAT, SGFN,
  and Open3DSG score dumps to one identity-preserving prediction schema.
- Open3DSG source-model support: `configure_open3dsg.py`,
  `prepare_open3dsg_splits.py`, and `select_open3dsg_checkpoint.py` implement
  source revision, data coverage, and development-loss selection checks.
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

## Source-prediction adapters

RelCompat3D consumes fixed relation predictions while retaining the scan,
3DSSG context, ordered instance pair, predicate, and source score. Run each
source predictor in its official environment and save one JSON object per line
with the fields below.

| Source | Required JSONL fields |
| --- | --- |
| VL-SAT | `scan_id`, `subset_split_id`, `subgraph_id`, `node_instance_ids`, `edge_indices`, `relation_names`, `rel_scores_3d` |
| SGFN | `scan_id`, `node_instance_ids`, `edge_indices`, `relation_names`, `rel_scores` |
| Open3DSG | `scan_id`, `subset_split_id`, `subgraph_id`, `edge`, `edge_index`, `predicate_scores` |

VL-SAT and SGFN score rows must follow the order in `relation_names`. SGFN
outputs cover a full scan and are mapped to official 3DSSG contexts by ordered
instance identity. Each Open3DSG `predicate_scores` entry contains the
predicate label, score, and predicate indices exported by Open3DSG.

Place the raw outputs at:

```text
local_dataset/RelCompat3D/source_outputs/
├── vlsat/raw.jsonl
├── sgfn/raw.jsonl
└── open3dsg/raw.jsonl
```

Run the adapters in the RelCompat3D container:

```bash
compose="docker compose -f configs/relcompat3d/compose.yaml"

env UID=$(id -u) GID=$(id -g) $compose run --rm relcompat3d_adapt_vlsat
env UID=$(id -u) GID=$(id -g) $compose run --rm relcompat3d_adapt_sgfn
env UID=$(id -u) GID=$(id -g) $compose run --rm relcompat3d_adapt_open3dsg
```

The adapters write:

```text
local_dataset/RelCompat3D/canonical/
├── vlsat/predictions.jsonl
├── sgfn/predictions.jsonl
└── open3dsg/predictions.jsonl
```

Each adjacent `predictions.manifest.json` records input and output hashes, row
counts, context counts, and identity checks. The adapters neither normalize
source scores nor create missing edges.

After restriction to the evaluated relation families, the frozen paper inputs
contain 220,848 VL-SAT rows, 159,444 Open3DSG rows, and 220,848 SGFN rows. The
adapters reproduce these counts, ordered-pair identities, and source scores.
Open3DSG hexadecimal split suffixes are converted to the integer split
identifiers used by 3DSSG.

Generate Open3DSG predictions with the checkpoint selected by the
[provided training configuration](../../configs/open3dsg/README.md), and
record its SHA-256 digest with the prediction run.

### Geometry join

The files above contain canonical source predictions. The corresponding
`verification.jsonl` files add ordered-pair measurements and frozen verifier
outputs. Build this join from officially obtained 3RScan/3DSSG data with the
thresholds in the frozen protocols. Changing those thresholds creates a new
evaluation rather than an exact reproduction of the reported experiment.
