# Open3DSG training configuration

This directory records the Open3DSG source-model training protocol used to
generate fixed open-vocabulary relation predictions for RelCompat3D.

- `protocol.json` records the source revision, data coverage, feature settings,
  training hyperparameters, and checkpoint-selection rule.
- `development_scans.txt` fixes the 30-scene development subset.
- `Dockerfile` defines the CUDA and Python environment.
- `compose.yaml` provides data preparation, feature extraction, training, and
  checkpoint-selection services.

The Open3DSG source code and 3RScan/3DSSG data remain external. Follow
[`docs/open3dsg-training.md`](../../docs/open3dsg-training.md) from the
repository root.
