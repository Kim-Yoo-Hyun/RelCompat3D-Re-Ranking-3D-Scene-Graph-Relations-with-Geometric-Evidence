# Support/Contact Applicability Routing

Status: `completed`

Development selection: `family_slot_rerank`

| Source | Method | R@100 | V@100 | dR vs source | dV vs source | support/contact exact |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vlsat | family-slot rerank | 0.9658 | 0.0295 | +0.0023 | -0.0182 | True |
| open3dsg | family-slot rerank | 0.5735 | 0.0332 | +0.0574 | -0.0910 | True |
| sgfn | family-slot rerank | 0.9303 | 0.0350 | +0.0068 | -0.0280 | True |

The selected route preserves the source-ranked family composition at every K and leaves support/contact selections unchanged; only proximity and vertical candidates are reordered within their source family slots.
