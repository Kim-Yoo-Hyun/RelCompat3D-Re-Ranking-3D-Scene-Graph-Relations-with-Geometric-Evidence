# Models and checkpoints

## RelCompat3D fitted models

The public model archive is:

- Google Drive: https://drive.google.com/file/d/1DaZoibKFyPS681e728Tzs613qscMgv4u/view
- filename: `relcompat3d_models_3dssg_v1.tar.zst`
- size: 36,796 bytes
- SHA-256: `4659858da8ff53f2c09769527ac486d182eef6e35c12ebeacfcc5d7ff6fdc103`

Restore and verify it with:

```bash
scripts/download_models.sh
scripts/validate.sh --require-models
```

The archive contains the eight JSON parameter files listed in
`configs/model_files.sha256`. The download script maps the archive experiment
paths to the current repository directories and verifies every model after
extraction.

## Source predictors

VL-SAT and SGFN use the checkpoints distributed or documented by their
official repositories:

- VL-SAT: https://github.com/wz7in/CVPR2023-VLSAT
- SGFN: https://github.com/ShunChengWu/3DSSG

Open3DSG source predictions are generated with the official implementation at
the revision recorded in `configs/open3dsg/protocol.json`. Train and select the
source checkpoint with:

```bash
scripts/train_open3dsg.sh prepare
scripts/train_open3dsg.sh preprocess
scripts/train_open3dsg.sh features
scripts/train_open3dsg.sh train
scripts/train_open3dsg.sh select
```

The protocol uses non-averaged BLIP features and selects the checkpoint with
the lowest development `val/loss`. The selection record writes the generated
file path, size, and SHA-256 digest to
`local_dataset/Open3DSG/selected_checkpoint.json`. See
`docs/open3dsg-training.md` for the complete procedure.

## License boundary

The RelCompat3D model archive contains only lightweight compatibility-model
parameters. It does not contain VL-SAT, SGFN, Open3DSG, or foundation-model
weights. Source-predictor model licenses remain those of their upstream
projects.
