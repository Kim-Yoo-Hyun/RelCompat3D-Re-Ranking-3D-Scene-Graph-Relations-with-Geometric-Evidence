# Reproduction guide

This guide separates fast paper-table regeneration from full source inference.
All reported experiment implementations run through Docker.

## A. Validate a Git checkout

```bash
scripts/validate.sh
```

This checks Compose parsing, Python compilation, JSON syntax, the active
artifact map, and the 291 stored canonical table cells. Missing external models
are reported but do not fail this Git-only check.

## B. Restore learned compatibility models

```bash
scripts/download_models.sh
scripts/validate.sh --require-models
```

The download script verifies both the archive and each extracted model file.

## C. Regenerate paper tables from local rows

Generate the local rows from official inputs as described in `docs/data.md`,
then run:

```bash
scripts/reproduce_tables.sh
```

The output directory is
`experiments/RelCompat3D_geom_reliability/paper_reproduction/regenerated/`.
The command creates Tables 1--3 as CSV and LaTeX, Figure 3 data and renderings,
and `canonical_validation.csv`. A valid reproduction reports 291 passing cells
with maximum absolute error no larger than `1e-12`.

For a maintainer recovery archive that uses the frozen pre-release directory
names, restore and remap it with:

```bash
scripts/restore_private_bundle.sh /path/to/RelCompat3D_AAAI27_release_20260730 core
scripts/reproduce_tables.sh
```

Use `all` instead of `core` to also restore the training/development and
point/mesh inputs. The script verifies the bundle manifest before writing the
current public paths.

## D. Refit and evaluate RelCompat3D

After all protocol inputs are mounted at their expected paths:

```bash
docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_fit
docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_freeze_initial
scripts/run_pipeline.sh initial
scripts/run_pipeline.sh downstream
```

The wrapper does not overwrite a nonempty incomplete output. It skips a stage
whose manifest already records `completed`.

## E. Supplementary analyses

Each command below uses a frozen protocol and writes into its corresponding
experiment directory. Public protocol paths use the `local_dataset/` layout
while retaining the input content hashes from the reported runs.

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

## F. Source-predictor inference

Source inference is intentionally not merged into the RelCompat3D container.
Run VL-SAT, SGFN, and Open3DSG with their official environments, then use the
three `relcompat3d_adapt_*` Docker services documented in
`docs/source-adapters.md`. This keeps third-party dependencies and licenses
separate while making the raw-score-to-prediction conversion explicit. Record
the exact source-predictor versions and checkpoints alongside newly generated
rows. For Open3DSG, use the pinned source revision, non-averaged BLIP
configuration, and development-loss selection procedure in
`docs/open3dsg-training.md`.

## Expected outputs

| Task | Output |
| --- | --- |
| Model fitting | `main_experiment/fit/` |
| Main evaluation | `main_experiment/evaluation/` |
| Table regeneration | `paper_reproduction/regenerated/` |
| Candidate oracle | `candidate_oracle/regenerated/` |
| Compact result index | `results/relcompat3d_geom_reliability/manifest.json` |

## What a fresh server still needs

- Docker with Compose v2
- official 3RScan/3DSSG access
- source-predictor environments and checkpoints for source inference
- the public RelCompat3D model archive
- canonical geometry and prediction inputs prepared locally from the official
  data and source-predictor repositories

The Git repository alone intentionally cannot reconstruct licensed geometry or
third-party predictions.
