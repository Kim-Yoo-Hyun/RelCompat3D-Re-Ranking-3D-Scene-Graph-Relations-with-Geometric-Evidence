# Scripts

- `download_models.sh`: download and verify the external RelCompat3D model archive.
- `validate.sh`: validate Compose, Python sources, JSON artifacts, model hashes,
  and frozen table cells.
- `reproduce_tables.sh`: regenerate Tables 1--3 and Figure 3 from table rows
  created locally from official data and source-predictor outputs.
- `run_pipeline.sh`: run the guarded main fitting/evaluation stages.
- `validate_repository.py`: standard-library repository and numerical checks.

All paper experiment execution remains Docker-based. Shell scripts only
coordinate pinned Compose services.
