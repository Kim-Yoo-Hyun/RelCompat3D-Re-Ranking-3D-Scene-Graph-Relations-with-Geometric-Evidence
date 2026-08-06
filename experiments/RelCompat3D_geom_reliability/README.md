# RelCompat3D experiments

This directory contains the active method and the analyses reported in the
paper and supplement.

| Directory | Purpose |
| --- | --- |
| `main_experiment/` | active fitting and main evaluation protocols |
| `paper_reproduction/` | Tables 1--3 and Figure 3 regeneration |
| `score_robustness/` | source-score mappings and simple baselines |
| `routing_controls/` | family-aware routing controls |
| `measurement_audit/` | measurement-dependence evidence index |
| `component_analysis/` | pairwise-loss and transformation checks |
| `seed_robustness/` | five-seed fitting analysis |
| `candidate_oracle/` | fixed-candidate Recall upper bounds |
| `factor_controls/` | factor-separation protocol and locks |
| `training_protocol/` | split firewall and train-only provenance |

`evaluation/` directories are frozen references. Reproduction services write
to ignored `regenerated/` directories. Fitted JSON models are restored with
`scripts/download_models.sh`.

The active evaluation uses 157 validation scans, 548 contexts, and 3,972
exact-match ground-truth relations. The model archive, licensed inputs, and
commands are described below.

## Data and runtime layout

Obtain 3RScan, 3DSSG, VL-SAT, SGFN, and Open3DSG from the official sources
linked in the repository README. Preserve their licenses and access
requirements. The frozen protocols expect the following local layout:

```text
local_dataset/RelCompat3D/
├── 3DSSG_subset/
│   ├── relationships.txt
│   ├── relationships_train.json
│   └── relationships_validation.json
├── source_outputs/
│   ├── vlsat/raw.jsonl
│   ├── open3dsg/raw.jsonl
│   └── sgfn/raw.jsonl
├── canonical/
│   ├── ground_truth.jsonl
│   ├── vlsat/{predictions,verification}.jsonl
│   ├── open3dsg/{predictions,verification}.jsonl
│   └── sgfn/{predictions,verification}.jsonl
└── secrets/
    └── table_rows_hmac_key.txt
```

After running the official predictor repositories, serialize their fixed
outputs as the three `raw.jsonl` files under `source_outputs/`. RelCompat3D
does not generate upstream source predictions.
The adapters create the three `predictions.jsonl` files. Each
`verification.jsonl` then adds measurements and frozen verifier labels from
the same ordered pair while retaining every source-prediction row and endpoint.
The [source adapter instructions](../../src/relcompat3d/README.md#source-prediction-adapters)
define the input contracts.

Training, development, geometry, verifier, and point/mesh paths are fixed in
`main_experiment/protocol.json` and `main_experiment/protocols/`. The
score-robustness, routing-control, and paper-reproduction protocols preserve
the frozen input hashes while mapping local files to this public layout.

## Model fitting

Model fitting requires the official `relationships_train.json`, the matching
3RScan OBB geometry, the tracked scan lists under `training_protocol/splits/`,
and the restored strict training model used to preserve the original feature
normalization contract. Restore the small model archive, regenerate the
training rows, and fit both estimators with:

```bash
scripts/download_models.sh
compose="docker compose -f configs/relcompat3d/compose.yaml"

$compose run --rm relcompat3d_build_training_rows
$compose run --rm relcompat3d_fit
$compose run --rm relcompat3d_fit_mlp
```

The training-row builder uses only the tracked 1,061 training scans and 117
internal-development scans. The 157 final-validation scans remain excluded.
Both fitting services use `--fit-only`, write to `main_experiment/regenerated/`,
and refuse nonempty output directories. The Linear and MLP outputs are
`fit/structured_models.json` and `nonlinear/models.json`, respectively.
Source-predictor evaluation is a separate step performed with restored or
newly fitted models after training is complete.

### Local paper rows

This exact paper-export stage also reads the point/mesh audit measurements
listed by `paper_reproduction/protocol.json` and a local HMAC key. These files
are not produced by the primary OBB/point verifier join and are not distributed
because they are derived from licensed scene geometry. Maintainers may restore
them from the private recovery archive. Other users can regenerate them from
officially obtained scans with the point/mesh audit code after accepting the
dataset terms.

Create the geometry-free inputs used by the paper-table script with:

```bash
docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_export_rows
```

The exporter writes:

```text
paper_reproduction/artifacts/table_rows/
├── ground_truth.csv.gz
├── open3dsg_candidates.csv.gz
├── sgfn_candidates.csv.gz
├── vlsat_candidates.csv.gz
└── schema.json
```

These local intermediates are derived from licensed annotations and source
predictions, so they are ignored by Git. Expected counts and hashes are stored
in `paper_reproduction/expected_rows.json`. The rows cover 157 scans, 548
contexts, and 3,972 ground-truth relations, with 220,848 VL-SAT candidates,
159,444 Open3DSG candidates, and 220,848 SGFN candidates.

The local HMAC key creates stable local identifiers without writing original
scan, context, instance, pair, or prediction identifiers to these rows. It does
not alter the terms of the source datasets. The exporter verifies the frozen
input hashes before writing any result.

## Reproduction commands

Validate a Git checkout:

```bash
scripts/validate.sh
```

Restore and verify the learned compatibility models:

```bash
scripts/download_models.sh
scripts/validate.sh --require-models
```

After exporting the local paper rows, regenerate Tables 1--3 and Figure 3:

```bash
scripts/reproduce_tables.sh
```

Outputs are written to `paper_reproduction/regenerated/`. A valid run reports
291 matching canonical values with maximum absolute error no larger than
`1e-12`.

Prepare the canonical inputs from official source outputs and evaluate the
restored models in a separate regenerated directory:

```bash
scripts/run_pipeline.sh prepare
scripts/download_models.sh
scripts/run_pipeline.sh evaluate
```

The frozen `main_experiment/evaluation/` directories are paper references and
are never overwritten by this route. Public evaluation writes to
`main_experiment/regenerated/public_evaluation/`. Exact re-fitting uses the
same `fit_linear.py` and `fit_mlp.py` implementations but additionally requires
the official training annotations and geometry, the listed split files, and
the restored strict normalization model described under Model fitting.

Supplementary analyses use the same frozen inputs:

```bash
compose="docker compose -f configs/relcompat3d/compose.yaml"

$compose run --rm relcompat3d_score_robustness
$compose run --rm relcompat3d_routing_constraints
$compose run --rm relcompat3d_measurement_audit
$compose run --rm relcompat3d_component_analysis
$compose run --rm relcompat3d_seed_robustness
$compose run --rm relcompat3d_candidate_oracle
$compose run --rm relcompat3d_runtime
```

Source-predictor inference remains in the official VL-SAT, SGFN, and Open3DSG
environments. Write their results to
`source_outputs/{vlsat,sgfn,open3dsg}/raw.jsonl`, then convert them with the
[source adapters](../../src/relcompat3d/README.md#source-prediction-adapters).
For Open3DSG, use the [pinned configuration](../../configs/open3dsg/README.md).

Expected output locations are:

| Task | Output |
| --- | --- |
| Restored reported models | `main_experiment/fit/` and `main_experiment/evaluation/nonlinear/` |
| Fresh main evaluation | `main_experiment/regenerated/public_evaluation/` |
| Fresh Linear fit | `main_experiment/regenerated/fit/` |
| Fresh MLP fit | `main_experiment/regenerated/nonlinear/` |
| Table regeneration | `paper_reproduction/regenerated/` |
| Candidate oracle | `candidate_oracle/regenerated/` |
| Compact result index | `../../results/relcompat3d_geom_reliability/manifest.json` |

A fresh server needs Docker with Compose v2, official 3RScan/3DSSG access,
source-predictor environments and checkpoints, the public RelCompat3D model
archive, and canonical geometry and prediction inputs prepared from the
official resources. The Git repository does not contain licensed geometry or
third-party predictions.

Maintainers with a verified recovery archive may restore the table inputs or
the complete local experiment state with:

```bash
scripts/restore_recovery_archive.sh /path/to/recovery-archive tables
scripts/restore_recovery_archive.sh /path/to/recovery-archive complete
```

The script verifies the archive manifest before writing files. This recovery
route is optional and is not required when inputs are generated from the
official resources.
