# Source-prediction adapters

RelCompat3D consumes fixed relation predictions with the identity of the scan,
3DSSG context, ordered instance pair, predicate, and source score intact. Run
the source predictors in their official environments, save the score fields
listed below, and run the adapters in the RelCompat3D Docker image.

## Raw score contracts

| Source | Required JSONL fields |
| --- | --- |
| VL-SAT | `scan_id`, `subset_split_id`, `subgraph_id`, `node_instance_ids`, `edge_indices`, `relation_names`, `rel_scores_3d` |
| SGFN | `scan_id`, `node_instance_ids`, `edge_indices`, `relation_names`, `rel_scores` |
| Open3DSG | `scan_id`, `subset_split_id`, `subgraph_id`, `edge`, `edge_index`, `predicate_scores` |

For VL-SAT and SGFN, every score row must follow the order in
`relation_names`. SGFN rows are full-scan outputs and are projected into the
official 3DSSG contexts by ordered instance identity. Open3DSG
`predicate_scores` entries contain `predicate_label`, `score`, and the
predicate indices exported by the official implementation.

Store the raw dumps at:

```text
local_dataset/RelCompat3D/source_outputs/
├── vlsat/raw.jsonl
├── sgfn/raw.jsonl
└── open3dsg/raw.jsonl
```

## Docker commands

```bash
compose="docker compose -f configs/relcompat3d/compose.yaml"

env UID=$(id -u) GID=$(id -g) $compose run --rm relcompat3d_adapt_vlsat
env UID=$(id -u) GID=$(id -g) $compose run --rm relcompat3d_adapt_sgfn
env UID=$(id -u) GID=$(id -g) $compose run --rm relcompat3d_adapt_open3dsg
```

The adapters write:

```text
local_dataset/RelCompat3D/canonical/
├── vlsat/predictions.jsonl
├── sgfn/predictions.jsonl
└── open3dsg/predictions.jsonl
```

Each adjacent `predictions.manifest.json` records the input and output hashes,
row counts, context counts, and identity checks. The adapter does not normalize
source scores or synthesize missing edges.

The adapters were checked against the frozen paper inputs after restricting
the output to support/contact, proximity, and vertical-order predicates. They
reproduce 220,848 VL-SAT rows, 159,444 Open3DSG rows, and 220,848 SGFN rows
with the same ordered-pair identities and source scores. Open3DSG hexadecimal
split suffixes are normalized to the integer split identifiers in 3DSSG.

Generate Open3DSG source predictions with the checkpoint selected according to
`docs/open3dsg-training.md`. Record its SHA-256 digest in the prediction-run
metadata before adapting the raw score dump.

## Geometry-join boundary

The files above are canonical source-prediction rows. The final
`verification.jsonl` files additionally contain ordered-pair measurements and
the frozen verifier output. That geometry join must use officially obtained
3RScan/3DSSG data and the thresholds recorded by the frozen protocols. Do not
substitute newly tuned thresholds and call the result an exact reproduction.
