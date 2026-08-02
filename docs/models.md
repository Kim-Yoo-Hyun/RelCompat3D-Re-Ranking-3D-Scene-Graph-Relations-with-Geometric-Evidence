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
`configs/model_files.sha256`. The download script maps the frozen archive
paths to the concise public experiment directories and verifies every model
after extraction.

## Source predictors

VL-SAT and SGFN use the checkpoints distributed or documented by their
official repositories:

- VL-SAT: https://github.com/wz7in/CVPR2023-VLSAT
- SGFN: https://github.com/ShunChengWu/3DSSG

Open3DSG source predictions use the official implementation with the selected
author-trained checkpoint. Its expected path is:

```text
local_dataset/Open3DSG_staged/training_repro/mlops/opensg/mlflow/
363094050435167554/25da9c4c00214f3b880cedbb2a124177/checkpoints/
epoch=13-step=13104.ckpt
```

The selected checkpoint is stored separately at:

https://drive.google.com/file/d/1PJNduscoRAB6cQcggBOo-ErzkiBs_QDG/view?usp=sharing

The shared file is named `epoch=13-step=13104.ckpt` and has size
419,735,447 bytes. Its SHA-256 digest is
`c1302882da43a7b985c10dd4f50177d5161ff6619090cd0eb4d5ff0411d64511`.
Restore and verify it at the path above with:

```bash
scripts/download_open3dsg_checkpoint.sh
```

The checkpoint is not needed for table regeneration from locally prepared
fixed prediction rows.

## License boundary

The RelCompat3D model archive contains only lightweight compatibility-model
parameters. It does not contain VL-SAT, SGFN, Open3DSG, or foundation-model
weights. Source-predictor checkpoint licenses remain those of their upstream
projects.
