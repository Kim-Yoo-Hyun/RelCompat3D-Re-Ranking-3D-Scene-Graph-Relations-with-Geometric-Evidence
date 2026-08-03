# Data and prediction inputs

RelCompat3D operates on fixed relation candidates and reconstructed geometry.
The repository does not redistribute 3RScan/3DSSG data or source-predictor
outputs. Obtain them from their official projects and follow their terms.
The raw score contracts and Docker adapter commands are in
[`docs/source-adapters.md`](source-adapters.md).

## Official sources

| Resource | Required content | Official source |
| --- | --- | --- |
| 3RScan | scans, point clouds, reconstructed meshes | https://github.com/WaldJohannaU/3RScan |
| 3DSSG | relationship annotations and official splits | https://3dssg.github.io/ |
| VL-SAT | model environment and fixed predictions | https://github.com/wz7in/CVPR2023-VLSAT |
| SGFN | model environment and fixed predictions | https://github.com/ShunChengWu/3DSSG |
| Open3DSG | official source, source-model training, and fixed predictions | https://github.com/boschresearch/Open3DSG |

The repository provides the Open3DSG configuration and checkpoint-selection
procedure in [`docs/open3dsg-training.md`](open3dsg-training.md).

## Runtime layout

The frozen protocols use paths below `local_dataset/RelCompat3D/`. A complete
full-evaluation setup provides at least:

```text
local_dataset/RelCompat3D/
├── 3DSSG_subset/
│   ├── relationships.txt
│   └── relationships_validation.json
├── source_outputs/
│   ├── vlsat/raw.jsonl
│   ├── open3dsg/raw.jsonl
│   └── sgfn/raw.jsonl
├── canonical/
│   ├── ground_truth.jsonl
│   ├── vlsat/{predictions,verification}.jsonl
│   ├── open3dsg/{predictions,verification}.jsonl
│   └── sgfn/{predictions,verification}.jsonl
└── secrets/
    └── table_rows_hmac_key.txt
```

The main fitting protocols additionally specify training and development rows,
geometry features, verifier rows, and point/mesh measurements. Their exact
paths are recorded in
`experiments/RelCompat3D_geom_reliability/main_experiment/protocol.json`
and its `protocols/` directory.

The score-robustness, routing-control, and table-reproduction protocols preserve
the frozen input hashes while mapping local paths to the public
`local_dataset/` layout.

The Docker adapter services produce the three `predictions.jsonl` files. The
`verification.jsonl` files add measurements and frozen verifier labels from
the corresponding ordered pair. They must preserve every source-prediction row
and its ordered endpoints.

## Local table rows

`relcompat3d_export_rows` converts locally prepared canonical inputs into
geometry-free rows used by the paper-table scripts. These files are local
intermediates, not a separate dataset download. The expected files are:

```text
experiments/RelCompat3D_geom_reliability/paper_reproduction/artifacts/table_rows/
├── ground_truth.csv.gz
├── open3dsg_candidates.csv.gz
├── sgfn_candidates.csv.gz
├── vlsat_candidates.csv.gz
└── schema.json
```

Expected counts and SHA-256 values are stored in
`paper_reproduction/expected_rows.json`. Because the rows are derived from
3RScan/3DSSG annotations and source-predictor outputs, they remain ignored and
are generated locally from officially obtained inputs:

```bash
docker compose -f configs/relcompat3d/compose.yaml run --rm relcompat3d_export_rows
```

The HMAC key is local-only and creates stable local identifiers without writing
the original scan, context, pair, instance, or prediction identifiers into the
table rows. It is not a substitute for the terms of the source datasets.

## Integrity checks

The exporter checks the input SHA-256 values in the frozen protocol. The local
table rows contain 548 contexts, 157 scans, 3,972 ground-truth relations, and
the following candidate counts:

| Predictor | Candidate rows |
| --- | ---: |
| VL-SAT | 220,848 |
| Open3DSG | 159,444 |
| SGFN | 220,848 |

Do not replace a hash-locked input with a different preprocessing result and
interpret the resulting numbers as a reproduction of the reported experiment.
