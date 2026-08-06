# Configuration

- `relcompat3d/` contains the canonical RelCompat3D Docker image and Compose
  services.
- `open3dsg/` contains the pinned Open3DSG source-model training protocol.
- `model_files.sha256` verifies the optional separately distributed reported
  model files. Fresh training does not require that archive.

```bash
docker compose -f configs/relcompat3d/compose.yaml config --quiet
docker compose -f configs/open3dsg/compose.yaml config --quiet
docker compose -f configs/relcompat3d/compose.yaml build relcompat3d_reproduce_rows
```
