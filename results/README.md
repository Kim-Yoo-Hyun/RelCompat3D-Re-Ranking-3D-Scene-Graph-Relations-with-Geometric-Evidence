# Results

`relcompat3d/` provides a compact index and summary of the reported results.
Detailed CSV and JSON outputs remain in the corresponding experiment folders.

| Item | Location |
| --- | --- |
| Tables 1--3 and Figure 3 data | `../experiments/relcompat3d/paper_results/evaluation/` |
| Method configuration | `../experiments/relcompat3d/method_config.json` |
| Main and supplementary analyses | `../experiments/relcompat3d/` |
| Result file index | `relcompat3d/manifest.json` |
| Result summary | `relcompat3d/report.md` |

The evaluation uses fixed VL-SAT, Open3DSG, and SGFN predictions on the shared
3DSSG validation split. It reports exact-match Recall and verifier-derived
Violation for `K in {5, 10, 20, 50, 100}`.
