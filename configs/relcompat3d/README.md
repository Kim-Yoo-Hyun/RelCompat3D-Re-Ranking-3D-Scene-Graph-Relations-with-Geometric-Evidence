# Docker environment

- `Dockerfile`: Python 3.11.9 runtime with dependencies from the root
  `requirements.lock.txt`.
- `compose.yaml`: fitting, evaluation, audit, robustness, row-export, and
  table-regeneration services.

The repository root is mounted at `/workspace`. Dataset and result locations
therefore follow the same relative paths on the host and in the container.
Run all commands from the repository root.
