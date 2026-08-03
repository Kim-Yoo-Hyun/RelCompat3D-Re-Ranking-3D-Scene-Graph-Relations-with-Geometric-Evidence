# Open3DSG source-model training

RelCompat3D uses fixed Open3DSG relation predictions. Generate the source
checkpoint from the official implementation and official 3RScan/3DSSG inputs
with the configuration in `configs/open3dsg/protocol.json`.

## 1. Obtain the official resources

Clone Open3DSG and check out the fixed revision:

```bash
git clone https://github.com/boschresearch/Open3DSG external/Open3DSG
git -C external/Open3DSG checkout a568358d6bb718929aa9ff67b2dfdecc4a4c3261
```

Obtain 3RScan, 3DSSG, and the Open3DSG auxiliary model files from their
official distributions. Preserve their licenses and access requirements. The
local runtime layout is:

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

The Open3DSG README defines the source and purpose of these files. This
repository does not redistribute them.

## 2. Build and prepare

Build the source-model environment and stage the fixed training and
development annotations:

```bash
docker compose -f configs/open3dsg/compose.yaml build open3dsg_environment
scripts/train_open3dsg.sh prepare
```

The preparation service verifies the Open3DSG commit and configures its path
module through container environment variables. The fixed development subset
contains 30 scenes and 160 annotated subgraphs before preprocessing.

## 3. Preprocess and extract features

```bash
scripts/train_open3dsg.sh preprocess
scripts/train_open3dsg.sh features
```

The coverage gate expects 3,744 of 3,852 training subgraphs and 156 of 160
development subgraphs after the official preprocessing procedure. Feature
extraction uses OpenSeg and non-averaged BLIP features with three scales and
five frames. The wrapper records the generated feature directory for the
training stage.

## 4. Train and select the checkpoint

```bash
scripts/train_open3dsg.sh train
scripts/train_open3dsg.sh select
```

Training uses seed 42, 100 epochs, batch size 1, one GPU, mixed precision, and
four-step gradient accumulation. The protocol does not enable
`--avg_blip_emb`. Checkpoint selection uses the minimum development
`val/loss`; relation evaluation metrics do not participate in selection.

The selection stage writes:

```text
local_dataset/Open3DSG/selected_checkpoint.json
local_dataset/Open3DSG/selected.ckpt
```

The JSON record contains the selected path, development loss, file size, and
SHA-256 digest. Supply `local_dataset/Open3DSG/selected.ckpt` to the official
Open3DSG inference command, then convert the resulting score dump with the
adapter described in `docs/source-adapters.md`.

## Reproduction boundary

Open3DSG sets seed 42, while the upstream Trainer permits nondeterministic GPU
operations. A new training execution therefore follows the same data,
configuration, and selection protocol without requiring an identical binary
digest. Record the generated checkpoint digest and source revision with every
new prediction export.
