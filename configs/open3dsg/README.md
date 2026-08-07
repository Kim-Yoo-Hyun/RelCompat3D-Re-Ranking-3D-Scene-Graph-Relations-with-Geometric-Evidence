# Open3DSG training configuration

This directory records the Open3DSG source-model training protocol used to
generate fixed open-vocabulary relation predictions for RelCompat3D.

- `protocol.json` records the source revision, data coverage, feature settings,
  training hyperparameters, and checkpoint-selection rule.
- `development_scans.txt` fixes the 30-scene development subset.
- `Dockerfile` defines the CUDA and Python environment.
- `compose.yaml` provides data preparation, feature extraction, training, and
  checkpoint-selection services.

The Open3DSG source code, 3RScan/3DSSG data, and auxiliary model files remain
external and must be obtained from their official distributions.

## Official resources

Clone Open3DSG and check out the source revision used by the protocol:

```bash
git clone https://github.com/boschresearch/Open3DSG external/Open3DSG
git -C external/Open3DSG checkout a568358d6bb718929aa9ff67b2dfdecc4a4c3261
```

Place the officially obtained resources as follows:

```text
local_dataset/Open3DSG/
├── input/3DSSG_subset/
│   ├── relationships_train.json
│   └── relationships_validation.json
├── data/3RScan/
│   ├── 3DSSG_subset/
│   ├── <official scan directories>
│   ├── 3RScan.json
│   ├── classes.txt
│   ├── relationships.txt
│   ├── relationships_custom.txt
│   ├── obj_boxes_train_refined.json
│   └── obj_boxes_val_refined.json
└── output/checkpoints/
    ├── blip2_positional_embedding.pt
    ├── pointnet.pth
    ├── pointnet2_ulip.pt
    └── openseg/
```

The [Open3DSG repository](https://github.com/boschresearch/Open3DSG) describes
the origin and purpose of these files. This repository does not redistribute
them.

## Training procedure

Build the environment and stage the fixed training and development
annotations:

```bash
docker compose -f configs/open3dsg/compose.yaml build open3dsg_environment
scripts/train_open3dsg.sh prepare
```

The preparation service verifies the Open3DSG commit and configures the source
paths through container environment variables. The fixed development subset
contains 30 scenes and 160 annotated subgraphs before preprocessing.

Preprocess the data and extract features:

```bash
scripts/train_open3dsg.sh preprocess
scripts/train_open3dsg.sh features
```

The coverage check expects 3,744 of 3,852 training subgraphs and 156 of 160
development subgraphs after official preprocessing. Feature extraction uses
OpenSeg and non-averaged BLIP features with three scales and five frames.

Train and select the checkpoint:

```bash
scripts/train_open3dsg.sh train
scripts/train_open3dsg.sh select
```

Training uses seed 42, 100 epochs, batch size 1, one GPU, mixed precision, and
gradient accumulation over four steps. The protocol does not enable
`--avg_blip_emb`. Selection uses the minimum development `val/loss`; relation
evaluation metrics do not participate in checkpoint selection.

The selection stage writes:

```text
local_dataset/Open3DSG/selected_checkpoint.json
local_dataset/Open3DSG/selected.ckpt
```

The JSON record contains the selected path, development loss, file size, and
SHA-256 digest. Use `selected.ckpt` with the official Open3DSG inference
command, then convert the score dump with the
[source adapter](../../src/relcompat3d/README.md#source-prediction-adapters).

## Reproduction boundary

The configuration fixes seed 42, but the upstream trainer permits
nondeterministic GPU operations. A new run therefore follows the same data,
configuration, and checkpoint-selection protocol without requiring an
identical checkpoint digest. Record the source revision and generated digest
with each prediction export.
