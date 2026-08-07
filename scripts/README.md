# Scripts

- `run_pipeline.sh`: run data preparation, RelCompat3D training, evaluation,
  point/mesh analysis, and table generation. Use `full` for the complete
  workflow or a stage name for a partial run.
- `train_open3dsg.sh`: prepare, train, and select a checkpoint from the pinned
  Open3DSG implementation.
- `validate.sh`: check Compose files, Python syntax, JSON files, and reported
  numerical results.
- `validate_repository.py`: implement repository and numerical checks used by
  `validate.sh`.

All experiment stages run through Docker Compose. Generated outputs are written
to ignored `regenerated/` directories.
