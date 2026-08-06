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
to ignored `regenerated/` directories. The training services generate the
fitted JSON models, which are ignored by Git.

The active evaluation uses 157 validation scans, 548 contexts, and 3,972
exact-match ground-truth relations. Licensed inputs and commands are described
below.

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

Training, development, geometry, verifier, and point/mesh paths are defined in
`main_experiment/protocol.json` and `main_experiment/protocols/`. Frozen
protocols preserve the paper input hashes. The `public_*` protocols use the
same method and metric contracts with locally generated official inputs.

## Model fitting

Model fitting requires the official `relationships_train.json`, matching
3RScan geometry, and the tracked scan lists under `training_protocol/splits/`.
The pipeline first generates the base feature template and training-split
normalization statistics, then fits the Linear and MLP estimators:

```bash
compose="docker compose -f configs/relcompat3d/compose.yaml"

$compose run --rm relcompat3d_build_training_rows
$compose run --rm relcompat3d_fit_base
$compose run --rm relcompat3d_fit
$compose run --rm relcompat3d_fit_mlp
```

The training-row builder uses only the tracked 1,061 training scans and 117
internal-development scans. The 157 final-validation scans remain excluded.
The fitting services write to `main_experiment/regenerated/` and refuse
nonempty output directories. The Linear and MLP outputs are
`fit/structured_models.json` and `nonlinear/models.json`, respectively.
Source-predictor evaluation is a separate step performed with the fitted
RelCompat3D estimators after training is complete.

### Local paper rows

Table export reads point/mesh audit measurements and a local HMAC key. The
fresh public route generates the measurements from officially obtained 3RScan
surfaces and creates the key locally. Neither file is distributed.

For a fresh run, generate the audit and table inputs with:

```bash
scripts/run_pipeline.sh audit-trained
scripts/run_pipeline.sh tables-trained
```

The exporter writes:

```text
paper_reproduction/regenerated/public_rows/
├── ground_truth.csv.gz
├── open3dsg_candidates.csv.gz
├── sgfn_candidates.csv.gz
├── vlsat_candidates.csv.gz
└── schema.json
```

These local intermediates are derived from licensed annotations and source
predictions, so they are ignored by Git. The public protocol requires the
official 157 scans, 548 contexts, and 3,972 ground-truth relations. Candidate
counts are recorded from the supplied source outputs rather than assumed to
match a different upstream inference run.

The local HMAC key creates stable local identifiers without writing original
scan, context, instance, pair, or prediction identifiers to these rows. The
frozen exporter verifies paper-run input hashes. The public exporter records
local hashes and applies structural checks.

## Reproduction commands

Validate a Git checkout:

```bash
scripts/validate.sh
```

After exporting the local paper rows, regenerate Tables 1--3 and Figure 3:

```bash
scripts/reproduce_tables.sh
```

Outputs are written to `paper_reproduction/regenerated/`. A valid run reports
291 matching canonical values with maximum absolute error no larger than
`1e-12`.

The frozen `main_experiment/evaluation/` directories are paper references and
are never overwritten by the reproduction route. Fresh-model evaluation writes
to `main_experiment/regenerated/trained_evaluation/`. Fitting uses the same
`fit_linear.py` and `fit_mlp.py` implementations as the reported experiments
and requires the official training annotations, geometry, and listed split
files.

The complete fresh route is:

```bash
scripts/run_pipeline.sh full
```

The script also exposes the five stages individually for partial reruns and
debugging. The `full` stage is the canonical end-to-end RelCompat3D command.

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
| Fresh main evaluation | `main_experiment/regenerated/public_evaluation/` |
| Fresh base template | `main_experiment/regenerated/base/` |
| Fresh Linear fit | `main_experiment/regenerated/fit/` |
| Fresh MLP fit | `main_experiment/regenerated/nonlinear/` |
| Fresh trained-model evaluation | `main_experiment/regenerated/trained_evaluation/` |
| Fresh point/mesh audit | `main_experiment/regenerated/public_surface_audit/` |
| Fresh table regeneration | `paper_reproduction/regenerated/public_tables/` |
| Frozen-value table check | `paper_reproduction/regenerated/` |
| Candidate oracle | `candidate_oracle/regenerated/` |
| Compact result index | `../../results/relcompat3d_geom_reliability/manifest.json` |

A fresh server needs Docker with Compose v2, official 3RScan/3DSSG access, and
the three source-predictor environments. Users train the source checkpoints in
the official VL-SAT, SGFN, and Open3DSG repositories and run inference to
create the three documented `raw.jsonl` files. The RelCompat3D training route
then generates the compatibility models and all subsequent evaluation and
table artifacts. The Git repository does not contain licensed geometry,
third-party checkpoints, or source predictions.
