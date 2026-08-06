# Scripts

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
  tables without overwriting frozen references. After source-predictor
  checkpoints and fixed outputs have been created in the official projects,
  `scripts/run_pipeline.sh full` invokes the complete RelCompat3D-owned route.
- `validate_repository.py`: standard-library repository and numerical checks.

All paper experiment execution remains Docker-based. Shell scripts only
coordinate pinned Compose services.

## Validation and additional analyses

`validate.sh` verifies the tracked code, configurations, protocols, and compact
result artifacts. When exact local table rows have been generated from the
official licensed inputs and corresponding source checkpoints,
`reproduce_tables.sh` regenerates Tables 1--3 and Figure 3 and checks the 291
reported values at tolerance `1e-12`.

Commands for the supplementary controls, robustness analyses, and maintainer
recovery workflow are documented in the
[experiment README](../experiments/RelCompat3D_geom_reliability/README.md#reproduction-commands).

The normal reproduction route trains the RelCompat3D estimators. The
`download_models.sh` and `restore_recovery_archive.sh` utilities are retained
only for maintainers who verify archived historical artifacts; they are not
part of the public training workflow.
