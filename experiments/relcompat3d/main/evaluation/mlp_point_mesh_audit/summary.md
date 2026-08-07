# RelCompat3D-MLP Surface Audit

Status: `completed`

The fixed point, mesh, and strict-consensus statuses are applied to RelCompat3D-MLP selections. Their absolute values are not directly comparable to the primary OBB-derived Violation metric.

| Predictor | K | Source consensus V | RelCompat3D-MLP consensus V | Change (95% scan-cluster CI) | MLP coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| vlsat | 5 | 0.0093 | 0.0042 | -0.0051 [-0.0095, -0.0017] | 0.9624 |
| vlsat | 10 | 0.0361 | 0.0279 | -0.0083 [-0.0127, -0.0046] | 0.9638 |
| vlsat | 20 | 0.0727 | 0.0617 | -0.0110 [-0.0143, -0.0078] | 0.9597 |
| vlsat | 50 | 0.1643 | 0.1405 | -0.0238 [-0.0273, -0.0207] | 0.9560 |
| vlsat | 100 | 0.2600 | 0.2157 | -0.0443 [-0.0484, -0.0410] | 0.9535 |
| open3dsg | 5 | 0.8320 | 0.1684 | -0.6636 [-0.6857, -0.6378] | 0.9872 |
| open3dsg | 10 | 0.6518 | 0.1318 | -0.5201 [-0.5405, -0.4998] | 0.9862 |
| open3dsg | 20 | 0.5380 | 0.0917 | -0.4463 [-0.4641, -0.4278] | 0.9856 |
| open3dsg | 50 | 0.4596 | 0.0884 | -0.3712 [-0.3896, -0.3528] | 0.9834 |
| open3dsg | 100 | 0.4087 | 0.1232 | -0.2856 [-0.2996, -0.2712] | 0.9815 |
| sgfn | 5 | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] | 0.8983 |
| sgfn | 10 | 0.0242 | 0.0132 | -0.0111 [-0.0227, -0.0021] | 0.9402 |
| sgfn | 20 | 0.0533 | 0.0355 | -0.0178 [-0.0254, -0.0114] | 0.9585 |
| sgfn | 50 | 0.1156 | 0.0763 | -0.0393 [-0.0448, -0.0343] | 0.9580 |
| sgfn | 100 | 0.2155 | 0.1588 | -0.0567 [-0.0615, -0.0522] | 0.9592 |

The audit is an automatic raw-surface construct check, not human physical-validity ground truth.
