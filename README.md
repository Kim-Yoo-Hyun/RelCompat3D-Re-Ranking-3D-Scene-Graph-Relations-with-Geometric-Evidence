# RelCompat3D

RelCompat3D learns predicate--geometry compatibility and uses it to re-rank
fixed 3D scene graph relation predictions. The repository provides Docker-based
code for training, family-aware re-ranking, evaluation, controls, point/mesh
analysis, and paper-table generation.

### [Project Page](https://kim-yoo-hyun.github.io/RelCompat3D-Re-Ranking-3D-Scene-Graph-Relations-with-Geometric-Evidence/)

![RelCompat3D method overview](site/assets/method.png)

RelCompat3D estimates compatibility from predicate semantics and ordered-pair
geometry without using the source relation score. Compatibility is combined
with that score only during re-ranking.

## Repository structure

```text
configs/       Docker environments and Compose services
experiments/   Experiment configurations and reported results
results/       Compact result index and summary
scripts/       Reproduction commands
site/          GitHub Pages project site
src/           Training, evaluation, analysis, and table-generation code
```

Licensed datasets, source-predictor checkpoints, predictions, and generated
RelCompat3D model files are not stored in Git.

## Setup

```bash
git clone https://github.com/Kim-Yoo-Hyun/RelCompat3D-Re-Ranking-3D-Scene-Graph-Relations-with-Geometric-Evidence.git
cd RelCompat3D-Re-Ranking-3D-Scene-Graph-Relations-with-Geometric-Evidence
docker compose -f configs/relcompat3d/compose.yaml build relcompat3d_generate_tables
scripts/validate.sh
```

The image uses Python 3.11.9 and the dependencies in `requirements.txt`.

## Data and source predictors

Obtain the datasets and predictor implementations from their official sources:

- [3RScan](https://github.com/WaldJohannaU/3RScan) and its
  [access page](https://waldjohannau.github.io/RIO/)
- [3DSSG](https://3dssg.github.io/)
- [VL-SAT](https://github.com/wz7in/CVPR2023-VLSAT)
- [SGFN/3DSSG](https://github.com/ShunChengWu/3DSSG)
- [Open3DSG](https://github.com/boschresearch/Open3DSG)

Train VL-SAT and SGFN with their official configurations. For Open3DSG, the
provided helper prepares the official implementation, extracts its features,
trains the model, and selects the checkpoint with the lowest development loss:

```bash
scripts/train_open3dsg.sh all
```

Run each predictor in its official environment and export fixed relation
predictions to:

```text
local_dataset/RelCompat3D/source_outputs/
├── vlsat/raw.jsonl
├── sgfn/raw.jsonl
└── open3dsg/raw.jsonl
```

The required fields are documented in the
[source-code README](src/relcompat3d/README.md#source-prediction-adapters).
RelCompat3D does not replace source-predictor training or inference.

## Reproduce the experiments

Prepare the official 3DSSG annotations under
`local_dataset/RelCompat3D/3DSSG_subset/` and the 3RScan scans under
`local_dataset/3RScan/scans/`. Then run:

```bash
scripts/run_pipeline.sh full
```

The command performs the following stages:

1. convert source predictions and join ordered-pair geometry;
2. build training examples and fit RelCompat3D-Linear and RelCompat3D-MLP;
3. evaluate both estimators on the fixed predictions;
4. run the point/mesh analysis; and
5. generate CSV files for Tables 1--3 and data/renderings for Figure 3.

The main outputs are written to:

```text
experiments/relcompat3d/training/calibration/regenerated/
experiments/relcompat3d/main/regenerated/{base,fit,mlp,evaluation}/
experiments/relcompat3d/main/regenerated/point_mesh_analysis/
experiments/relcompat3d/paper_results/regenerated/{rows,tables}/
```

Individual stages are available as `prepare`, `train`, `evaluate`, `audit`,
and `tables`. See the [experiment README](experiments/relcompat3d/README.md)
for the local data layout and supplementary analyses.

## Results

At `K=50`, the reported Recall/Violation percentages are:

| Predictor | Source | RelCompat3D-Linear | RelCompat3D-MLP |
| --- | ---: | ---: | ---: |
| VL-SAT | 92.72 / 2.68 | 92.77 / 1.97 | 92.72 / 1.89 |
| Open3DSG | 40.43 / 13.87 | 44.18 / 3.42 | 46.70 / 4.13 |
| SGFN | 74.02 / 3.85 | 74.50 / 2.63 | 74.57 / 2.58 |

These values cover support/contact, proximity, and vertical-order relations on
the shared 3DSSG validation split. Additional results are indexed in
[results/](results/README.md).

![Recall--Violation results across VL-SAT, Open3DSG, and SGFN](site/assets/results.png)

## Reproducibility scope

A Git checkout verifies the code, configurations, and reported result files.
Full training and evaluation additionally require officially obtained
3RScan/3DSSG data and fixed predictions produced by the official VL-SAT,
Open3DSG, and SGFN implementations. With these inputs,
`scripts/run_pipeline.sh full` covers the RelCompat3D workflow through table
generation. Dataset and predictor licenses remain with their respective
owners.

## Citation

The paper citation will be added after publication. Software metadata is
available in [CITATION.cff](CITATION.cff).

## License

RelCompat3D code is released under the [Apache License 2.0](LICENSE). See
[third-party licenses](third_party_licenses.md) for external resources.
