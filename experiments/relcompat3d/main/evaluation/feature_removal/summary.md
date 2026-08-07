# Held-out Geometry-Primitive Evaluation

Status: `completed`

All variants are refitted on the 1,061-scan training split, use no source score inside compatibility, retain exact orbit projection, and use the same family-slot route. Support/contact is unchanged.

## K=50 overall

| Source | Condition | Recall | verifier V | delta R vs source | delta V vs source |
| --- | --- | ---: | ---: | ---: | ---: |
| vlsat | Full RelCompat3D | 0.9277 | 0.0197 | +0.0005 | -0.0070 |
| vlsat | Exact verifier scalar held out | 0.9280 | 0.0199 | +0.0008 | -0.0069 |
| vlsat | Verifier primitive family held out | 0.9275 | 0.0268 | +0.0003 | +0.0000 |
| vlsat | Alternative evidence only | 0.9280 | 0.0268 | +0.0008 | +0.0000 |
| open3dsg | Full RelCompat3D | 0.4418 | 0.0342 | +0.0375 | -0.1045 |
| open3dsg | Exact verifier scalar held out | 0.4381 | 0.0342 | +0.0337 | -0.1045 |
| open3dsg | Verifier primitive family held out | 0.4119 | 0.0954 | +0.0076 | -0.0433 |
| open3dsg | Alternative evidence only | 0.4104 | 0.1194 | +0.0060 | -0.0193 |
| sgfn | Full RelCompat3D | 0.7450 | 0.0263 | +0.0048 | -0.0122 |
| sgfn | Exact verifier scalar held out | 0.7465 | 0.0268 | +0.0063 | -0.0117 |
| sgfn | Verifier primitive family held out | 0.7467 | 0.0388 | +0.0065 | +0.0003 |
| sgfn | Alternative evidence only | 0.7462 | 0.0386 | +0.0060 | +0.0002 |

The exact-scalar condition removes the normalized scalar consumed by the corresponding verifier. The primitive-family condition also removes raw or deterministically related measurements. The alternative-evidence condition retains overlap-only proximity evidence and horizontal-distance/overlap vertical context, so it cannot reconstruct the verifier's directed vertical scalar.
