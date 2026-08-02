# RelCompat3D Runtime Benchmark

Status: `completed`

The timing starts from preloaded verification rows and includes compatibility scoring, transformation averaging, and family-aware sorting. It excludes source prediction, geometry reconstruction/join, file parsing, metrics, and bootstrap.

| Predictor | Contexts | In-scope rows | Scored rows | Median total (s) | Median ms/context | Scored rows/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VL-SAT | 548 | 220,848 | 110,424 | 2.4447 | 4.461 | 45,170 |
| Open3DSG | 533 | 159,444 | 79,722 | 1.8075 | 3.391 | 44,105 |
| SGFN | 548 | 220,848 | 110,424 | 2.4528 | 4.476 | 45,019 |

Stored parameters: `66`; active proximity/vertical parameters: `43`; fitted fusion parameters: `0`.
Peak process RSS: `366.5 MiB`.
