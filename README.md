# RelCompat3D

RelCompat3D learns predicate--geometry compatibility and uses it to re-rank
fixed 3D scene graph relation predictions. The implementation covers model
fitting, family-aware re-ranking, controls, bootstrap evaluation, point/mesh
audits, and paper-table regeneration for VL-SAT, Open3DSG, and SGFN on 3DSSG.

### [Project Page](https://kim-yoo-hyun.github.io/RelCompat3D-Re-Ranking-3D-Scene-Graph-Relations-with-Geometric-Evidence/)

![RelCompat3D method overview](site/assets/method.png)

RelCompat3D estimates compatibility from predicate semantics and ordered-pair
geometry without the source relation score, then combines both signals during
family-aware re-ranking.

## Repository structure

```text
configs/       Pinned Docker environment and Compose services
experiments/   Frozen protocols and compact paper evidence
results/       Paper-facing result index
scripts/       Validation and experiment wrappers
site/          Static GitHub Pages project site
src/           Training, evaluation, audit, and table-generation code
```

Large datasets, source-predictor checkpoints, prediction rows, and trained
RelCompat3D parameter files are not stored in Git. Official sources and local
paths are specified below and in the README of the corresponding folder.

## 1. Setup

Docker is the canonical environment. The image uses Python 3.11.9 and the
fully pinned `requirements.lock.txt` environment.

```bash
git clone https://github.com/Kim-Yoo-Hyun/RelCompat3D-Re-Ranking-3D-Scene-Graph-Relations-with-Geometric-Evidence.git
cd RelCompat3D-Re-Ranking-3D-Scene-Graph-Relations-with-Geometric-Evidence
docker compose -f configs/relcompat3d/compose.yaml build relcompat3d_reproduce_rows
scripts/validate.sh
```

For local source inspection only:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q src/relcompat3d
```

## 2. Prepare official data and prediction rows

Obtain 3RScan/3DSSG and the source predictors from their official projects:

- [3RScan](https://github.com/WaldJohannaU/3RScan) and its
  [access page](https://waldjohannau.github.io/RIO/)
- [3DSSG](https://3dssg.github.io/)
- [VL-SAT](https://github.com/wz7in/CVPR2023-VLSAT)
- [SGFN/3DSSG](https://github.com/ShunChengWu/3DSSG)
- [Open3DSG](https://github.com/boschresearch/Open3DSG)

### Train the source predictors

Source-predictor checkpoints are required. They are not included in this
repository. Create them in the official predictor repositories before running
RelCompat3D:

- VL-SAT: prepare its 3RScan/3DSSG data and run
  `python -m main --mode train --config <config_path> --exp <exp_name>`.
- SGFN: prepare the official 3DSSG experiment data and run
  `python main.py --mode train --config <config_path>` with the 160-object,
  26-relation configuration used by this evaluation.
- Open3DSG: use the pinned helper below to train the official implementation
  with the non-averaged BLIP configuration and select the checkpoint with the
  lowest development loss.

```bash
scripts/train_open3dsg.sh prepare
scripts/train_open3dsg.sh preprocess
scripts/train_open3dsg.sh features
scripts/train_open3dsg.sh train
scripts/train_open3dsg.sh select
```

The source revision, data coverage gates, hyperparameters, and selection rule
are described [here](configs/open3dsg/README.md).

Run inference from each resulting checkpoint, then export its fixed relation
candidates to the JSONL contract below. The checkpoints remain in their
respective source repositories and are inputs to source inference, not inputs
to the RelCompat3D fitting code.

The repository does not redistribute licensed scans, meshes, annotations,
dataset-derived candidate rows, or third-party checkpoints. Run inference in
each official source-predictor repository, serialize its fixed relation scores
to the documented JSONL contract, and create exactly these files:

```text
local_dataset/RelCompat3D/source_outputs/
├── vlsat/raw.jsonl
├── sgfn/raw.jsonl
└── open3dsg/raw.jsonl
```

RelCompat3D does not invoke, replace, or synthesize the upstream inference
stage. With the three files above in place, run the complete public
preprocessing route:

```bash
scripts/run_pipeline.sh prepare
```

The command adapts the official outputs, exports exact-relation ground truth,
and joins each candidate with ordered-pair geometry and verifier outputs. The
raw schemas are described
[here](src/relcompat3d/README.md#source-prediction-adapters), and the official
dataset layout is described
[here](experiments/RelCompat3D_geom_reliability/README.md#data-and-runtime-layout).

## 3. Train, evaluate, and regenerate tables

Place `relationships_train.json` beside the validation annotation and make the
official 3RScan scans available under `local_dataset/3RScan/scans/`. After the
source checkpoints have produced the three upstream `raw.jsonl` files, the
following command trains RelCompat3D and runs the complete repository-owned
route:

```bash
scripts/run_pipeline.sh full
```

This command performs five stages in order:

1. adapt source-predictor outputs and join ordered-pair geometry;
2. build linked positive--counterfactual training rows and fit the base,
   Linear, and MLP estimators;
3. evaluate the freshly fitted models;
4. derive point- and mesh-based audit measurements from official 3RScan
   surfaces; and
5. generate CSV/LaTeX versions of Tables 1--3 and the Figure 3 data and
   renderings.

Fresh outputs are isolated under:

```text
experiments/RelCompat3D_geom_reliability/training_protocol/calibration/regenerated/
experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/{base,fit,nonlinear}/
experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/trained_evaluation/
experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_surface_audit/
experiments/RelCompat3D_geom_reliability/paper_reproduction/regenerated/{public_rows,public_tables}/
```

The source checkpoints are used only to produce the three input files at this
stage. RelCompat3D then fits its compatibility estimators from the official
training split. The tracked 1,061/117/157 scan lists define the training,
internal-development, and final-validation firewall. The fitting commands do
not read source scores, predictor identities, source-predictor validation rows,
or final-validation labels. See the
[experiment README](experiments/RelCompat3D_geom_reliability/README.md#model-fitting)
for individual Docker commands and required inputs.

The generated tables reflect the source predictions supplied by the user.
They match the paper values only when the same official predictor
configurations, checkpoints, and fixed candidate outputs are used. The public
route checks dataset scope, identities, ranking constraints, metric accounting,
and output integrity without treating a different upstream run as a failed
paper-value comparison.

## 4. Frozen-result checks

`scripts/validate.sh` verifies the tracked code, configurations, protocols, and
compact result artifacts. If the exact local table rows for the reported run
have been generated from the official licensed inputs and the corresponding
source checkpoints, `scripts/reproduce_tables.sh` regenerates the paper tables
and checks 291 frozen values at tolerance `1e-12`.

## 5. Extended analyses

Additional controls, robustness analyses, and maintainer recovery commands are listed
[here](experiments/RelCompat3D_geom_reliability/README.md#reproduction-commands).

## Results

Frozen compact outputs are tracked for inspection and integrity checks. The
main results are summarized below, and the complete artifact locations are
listed [here](results/README.md).

At `K=50`, the reported Recall/Violation percentages are:

| Predictor | Source | RelCompat3D-Linear | RelCompat3D-MLP |
| --- | ---: | ---: | ---: |
| VL-SAT | 92.72 / 2.68 | 92.77 / 1.97 | 92.72 / 1.89 |
| Open3DSG | 40.43 / 13.87 | 44.18 / 3.42 | 46.70 / 4.13 |
| SGFN | 74.02 / 3.85 | 74.50 / 2.63 | 74.57 / 2.58 |

These values use the scoped support/contact, proximity, and vertical-order
evaluation on the shared 3DSSG validation scenes.

![Recall--Violation results across VL-SAT, Open3DSG, and SGFN](site/assets/results.png)

## Reproducibility boundary

There are two supported levels:

1. A Git-only checkout validates code, configuration, protocols, and frozen
   results.
2. A Git checkout combined with officially obtained 3RScan/3DSSG data,
   source-predictor checkpoints trained in the official repositories, and the
   resulting fixed predictions supports RelCompat3D training, geometry
   joining, evaluation, point/mesh auditing, and table generation.

Source-predictor inference remains governed by the upstream licenses and data
terms. VL-SAT and SGFN follow their official repositories. The Open3DSG
training configuration in this repository invokes the pinned official source
without redistributing its code or model files.

The repository does not replace training or inference in the three independent
source-predictor repositories. Users first create the source checkpoints and
fixed predictions with those official projects. Once the predictions use the
documented contract, `scripts/run_pipeline.sh full` covers the complete
RelCompat3D-owned process through table generation.

## Citation

The paper citation will be added after publication. For the software release,
GitHub can read the metadata in [CITATION.cff](CITATION.cff).

## License

RelCompat3D code is released under the [Apache License 2.0](LICENSE). Dataset,
source-predictor, and checkpoint licenses remain with their respective owners;
see [third-party licenses](third_party_licenses.md).
