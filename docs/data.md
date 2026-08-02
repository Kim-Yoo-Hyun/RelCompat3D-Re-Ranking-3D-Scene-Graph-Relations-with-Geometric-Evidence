# Data and prediction inputs

RelCompat3D operates on fixed relation candidates and reconstructed geometry.
The repository does not redistribute 3RScan/3DSSG data or source-predictor
outputs. Obtain them from their official projects and follow their terms.

## Official sources

| Resource | Required content | Official source |
| --- | --- | --- |
| 3RScan | scans, point clouds, reconstructed meshes | https://github.com/WaldJohannaU/3RScan |
| 3DSSG | relationship annotations and official splits | https://3dssg.github.io/ |
| VL-SAT | model environment and fixed predictions | https://github.com/wz7in/CVPR2023-VLSAT |
| SGFN | model environment and fixed predictions | https://github.com/ShunChengWu/3DSSG |
| Open3DSG | model environment and fixed predictions | https://github.com/boschresearch/Open3DSG |

## Runtime layout

The frozen protocols use paths below `local_dataset/RelCompat3D/`. A complete
full-evaluation setup provides at least:

```text
local_dataset/RelCompat3D/
├── 3DSSG_subset/
│   └── relationships_validation.json
├── canonical/
│   ├── ground_truth.jsonl
│   ├── vlsat/verification.jsonl
│   ├── open3dsg/verification.jsonl
│   └── sgfn/verification.jsonl
└── secrets/
    └── row_reproduction_hmac_key.txt
```

The main fitting protocols additionally specify training and development rows,
geometry features, verifier rows, and point/mesh measurements. Their exact
paths are recorded in
`experiments/RelCompat3D_geom_reliability/no_family_indicator_v1/protocol.json`
and its `protocols/` directory.

The score-robustness, routing-control, and row-reproduction protocols preserve
the frozen input hashes while mapping local paths to the public
`local_dataset/` layout.

## Derived row bundle

`relcompat3d_export_rows` converts licensed canonical inputs into a geometry-free
evaluation bundle. The expected files are:

```text
experiments/RelCompat3D_geom_reliability/row_reproduction_v1/artifacts/derived_rows/
├── ground_truth.csv.gz
├── open3dsg_candidates.csv.gz
├── sgfn_candidates.csv.gz
├── vlsat_candidates.csv.gz
└── schema.json
```

Expected counts and SHA-256 values are stored in
`row_reproduction_v1/expected_bundle.json`. The bundle contains transformed
candidate records, so it is not publicly redistributed unless the applicable
3RScan/3DSSG terms or data owner explicitly permit it. Authorized users can
create it locally:

```bash
docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_export_rows
```

The HMAC key is local-only. It removes original benchmark identifiers from the
derived rows and is not required when reproducing tables from an existing
authorized bundle.

## Integrity checks

The exporter checks the input SHA-256 values in the frozen protocol. The paper
bundle contains 548 contexts, 157 scans, 3,972 ground-truth relations, and the
following candidate counts:

| Predictor | Candidate rows |
| --- | ---: |
| VL-SAT | 220,848 |
| Open3DSG | 159,444 |
| SGFN | 220,848 |

Do not replace a hash-locked input with a different preprocessing result and
interpret the resulting numbers as a reproduction of the reported experiment.
