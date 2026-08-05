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
scripts/       Validation, model restoration, and experiment wrappers
site/          Static GitHub Pages project site
src/           Training, evaluation, audit, and table-generation code
tests/         Synthetic source-adapter, geometry, verifier, and routing checks
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

## 2. Restore RelCompat3D models

The lightweight fitted models are available [here](https://drive.google.com/file/d/1DaZoibKFyPS681e728Tzs613qscMgv4u/view).
The following command downloads the 36 KB archive, verifies its SHA-256
digest, and extracts the fitted models to their expected experiment paths.

```bash
scripts/download_models.sh
scripts/validate.sh --require-models
```

## 3. Prepare data and prediction rows

Obtain 3RScan/3DSSG and the source predictors from their official projects:

- [3RScan](https://github.com/WaldJohannaU/3RScan) and its
  [access page](https://waldjohannau.github.io/RIO/)
- [3DSSG](https://3dssg.github.io/)
- [VL-SAT](https://github.com/wz7in/CVPR2023-VLSAT)
- [SGFN/3DSSG](https://github.com/ShunChengWu/3DSSG)
- [Open3DSG](https://github.com/boschresearch/Open3DSG)

For Open3DSG, train the pinned official implementation with the provided
non-averaged BLIP configuration and select the checkpoint with the lowest
development loss:

```bash
scripts/train_open3dsg.sh prepare
scripts/train_open3dsg.sh preprocess
scripts/train_open3dsg.sh features
scripts/train_open3dsg.sh train
scripts/train_open3dsg.sh select
```

The source revision, data coverage gates, hyperparameters, and selection rule
are described [here](configs/open3dsg/README.md).

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

Restore the lightweight RelCompat3D models and evaluate the locally prepared
rows without writing into frozen reference directories:

```bash
scripts/download_models.sh
scripts/run_pipeline.sh evaluate
```

Evaluation outputs are written to
`experiments/RelCompat3D_geom_reliability/main_experiment/regenerated/public_evaluation/`.

Exact Tables 1--3 and Figure 3 regeneration additionally requires the local
point/mesh audit measurements and HMAC key listed in the
[experiment README](experiments/RelCompat3D_geom_reliability/README.md#local-paper-rows).
After preparing those licensed-data derivatives, create the local paper rows
with:

```bash
scripts/run_pipeline.sh tables
```

The resulting local intermediate belongs at:

```text
experiments/RelCompat3D_geom_reliability/paper_reproduction/artifacts/table_rows/
```

## 4. Reproduce Tables 1--3 and Figure 3

To rerun table generation after the local rows already exist, use:

```bash
scripts/reproduce_tables.sh
```

Outputs are written to:

```text
experiments/RelCompat3D_geom_reliability/paper_reproduction/regenerated/
```

The command regenerates CSV and LaTeX versions of Tables 1--3, Figure 3 data
and renderings, and a cell-level comparison against 291 frozen paper values.
The accepted tolerance is `1e-12`; the frozen reference run has maximum
absolute error `0`.

## 5. Training and extended analyses

The repository includes the Linear and MLP fitting code, split files, frozen
training protocol, counterfactual construction, and component analyses. The
public commands above restore the reported fitted models and rerun evaluation
on rows prepared from official data. Re-fitting requires the official 3DSSG
training annotations and the train/internal-development inputs listed in the
[experiment README](experiments/RelCompat3D_geom_reliability/README.md); it is
kept separate from paper evaluation so a fresh run cannot overwrite frozen
evidence.

Additional controls, robustness analyses, and recovery commands are listed
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

There are three supported levels:

1. A Git-only checkout validates code, configuration, protocols, and frozen
   results.
2. Git plus the public RelCompat3D model archive restores all learned
   compatibility parameters.
3. A Git checkout combined with officially obtained 3RScan/3DSSG data and
   outputs generated with the official source-predictor repositories supports
   local geometry joining, evaluation, and paper-table regeneration. The
   fitting code and frozen training protocol are also included for controlled
   re-fitting with the corresponding official train/development inputs.

Source-predictor inference remains governed by the upstream licenses and data
terms. VL-SAT and SGFN follow their official repositories. The Open3DSG
training configuration in this repository invokes the pinned official source
without redistributing its code or model files.

The repository does not provide a single command from raw dataset download
through inference of all three source predictors; data access and
source-predictor inference follow their official repositories.

## Citation

The paper citation will be added after publication. For the software release,
GitHub can read the metadata in [CITATION.cff](CITATION.cff).

## License

RelCompat3D code is released under the [Apache License 2.0](LICENSE). Dataset,
source-predictor, and checkpoint licenses remain with their respective owners;
see [third-party licenses](third_party_licenses.md).
