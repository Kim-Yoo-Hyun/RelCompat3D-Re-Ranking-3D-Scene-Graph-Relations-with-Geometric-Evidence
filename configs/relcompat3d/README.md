# Docker environment

- `Dockerfile`: Python 3.11.9 runtime with dependencies from the root
  `requirements.lock.txt`.
- `compose.yaml`: source adaptation, geometry joining, fitting, evaluation,
  audit, robustness, row-export, table-regeneration, and synthetic-test
  services.

The repository root is mounted at `/workspace`. Dataset and result locations
therefore follow the same relative paths on the host and in the container.
Run all commands from the repository root.
