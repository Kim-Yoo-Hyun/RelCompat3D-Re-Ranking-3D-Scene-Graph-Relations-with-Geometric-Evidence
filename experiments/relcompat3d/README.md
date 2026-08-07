# RelCompat3D experiments

This directory contains the configurations and reported analyses used by the
paper and supplement.

| Directory | Purpose |
| --- | --- |
| `main/` | Linear/MLP training and main evaluation |
| `paper_results/` | Tables 1--3 and Figure 3 generation |
| `score_robustness/` | source-score transformations and simple baselines |
| `routing_controls/` | family-aware ranking controls |
| `measurement_analysis/` | measurement-source comparison |
| `component_analysis/` | pairwise-loss and transformation analyses |
| `seed_robustness/` | five-seed training analysis |
| `candidate_oracle/` | fixed-candidate Recall upper bounds |
| `training/` | data splits and training-row configuration |

Reported outputs are stored in `evaluation/`. New executions write to ignored
`regenerated/` directories.

## Data layout

Obtain 3RScan, 3DSSG, VL-SAT, SGFN, and Open3DSG from the official links in the
root README. Prepare the following local layout:

```text
local_dataset/
├── 3RScan/scans/
└── RelCompat3D/
    ├── 3DSSG_subset/
    │   ├── relationships.txt
    │   ├── relationships_train.json
    │   └── relationships_validation.json
    ├── source_outputs/
    │   ├── vlsat/raw.jsonl
    │   ├── open3dsg/raw.jsonl
    │   └── sgfn/raw.jsonl
    ├── prepared/
    └── secrets/
```

The three `raw.jsonl` files are produced by the official source-predictor
repositories. Their format is defined in
[src/relcompat3d/README.md](../../src/relcompat3d/README.md#source-prediction-adapters).

## Main pipeline

Run the complete RelCompat3D workflow with:

```bash
scripts/run_pipeline.sh full
```

The stages can also be run separately:

```bash
scripts/run_pipeline.sh prepare
scripts/run_pipeline.sh train
scripts/run_pipeline.sh evaluate
scripts/run_pipeline.sh audit
scripts/run_pipeline.sh tables
```

Training uses the 1,061 training scans. The 117 development scans are used for
diagnostics, and the 157 validation scans are used only for final evaluation.
The fitting code does not use source relation scores, predictor identity, or
validation labels.

Main generated outputs are:

| Stage | Output |
| --- | --- |
| Training rows | `training/calibration/regenerated/` |
| Base models | `main/regenerated/base/` |
| RelCompat3D-Linear | `main/regenerated/fit/` |
| RelCompat3D-MLP | `main/regenerated/mlp/` |
| Evaluation | `main/regenerated/evaluation/` |
| Point/mesh analysis | `main/regenerated/point_mesh_analysis/` |
| Tables and figure data | `paper_results/regenerated/tables/` |

## Supplementary analyses

After the required inputs are available, run individual analyses with Docker
Compose:

```bash
compose="docker compose -f configs/relcompat3d/compose.yaml"

$compose run --rm relcompat3d_score_robustness
$compose run --rm relcompat3d_routing_constraints
$compose run --rm relcompat3d_measurement_analysis
$compose run --rm relcompat3d_component_analysis
$compose run --rm relcompat3d_seed_robustness
$compose run --rm relcompat3d_candidate_oracle
$compose run --rm relcompat3d_runtime
```

These analyses use the same data splits, relation families, rank cutoffs, and
metric definitions as the main evaluation. Generated datasets, checkpoints,
source predictions, and model files remain outside Git.
