# Scripts

- `download_models.sh`: download and verify the external RelCompat3D model archive.
- `train_open3dsg.sh`: coordinate the pinned Open3DSG preparation, feature
  extraction, source-model training, and checkpoint-selection stages.
- `restore_recovery_archive.sh`: verify a maintainer recovery archive and map
  its contents to the current repository layout.
- `validate.sh`: validate Compose, Python sources, JSON artifacts, model hashes,
  and frozen table cells.
- `reproduce_tables.sh`: regenerate Tables 1--3 and Figure 3 from table rows
  created locally from official data and source-predictor outputs.
- `run_pipeline.sh`: adapt official source outputs, build geometry/verifier and
  training rows, fit both estimators, run the point/mesh audit, and generate
  tables without overwriting frozen references. `make from-official` invokes
  its complete repository-owned route.
- `validate_repository.py`: standard-library repository and numerical checks.

All paper experiment execution remains Docker-based. Shell scripts only
coordinate pinned Compose services.
