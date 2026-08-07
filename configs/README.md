# Configuration

- `relcompat3d/` contains the Docker image and Compose services for training,
  evaluation, analyses, and table generation.
- `open3dsg/` contains the pinned Open3DSG preparation and training setup.

Validate and build the RelCompat3D configuration with:

```bash
docker compose -f configs/relcompat3d/compose.yaml config --quiet
docker compose -f configs/open3dsg/compose.yaml config --quiet
docker compose -f configs/relcompat3d/compose.yaml build relcompat3d_generate_tables
```
