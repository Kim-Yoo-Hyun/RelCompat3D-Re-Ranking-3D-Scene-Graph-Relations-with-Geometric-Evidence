# Scan-Cluster Bootstrap Sensitivity

Status: `completed`

The promoted rankings and point estimates are unchanged. This sensitivity resamples 157 scans with replacement and carries every relation context from each sampled scan together.

| Source | dRecall@100 (95% scan-cluster CI) | dVerifier-V@100 (95% scan-cluster CI) |
| --- | ---: | ---: |
| VL-SAT | +0.0053 [+0.0004, +0.0114] | -0.0152 [-0.0172, -0.0135] |
| Open3DSG | +0.0891 [+0.0646, +0.1119] | -0.0902 [-0.0958, -0.0845] |
| SGFN | +0.0179 [+0.0132, +0.0223] | -0.0258 [-0.0285, -0.0233] |

At K=100, no Recall interval crosses below zero (the VL-SAT lower bound reaches zero), and all verifier-V intervals remain below zero. This is a dependence sensitivity, not a new score-selection result.
