# Configuration

`relcompat3d/` contains the canonical Docker image and Compose services.
`model_files.sha256` verifies the separately distributed fitted model files.

```bash
docker compose -f configs/relcompat3d/compose.yaml config --quiet
docker compose -f configs/relcompat3d/compose.yaml build relcompat3d_reproduce_rows
```
