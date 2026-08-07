# Construct-Dependence Evidence Package

Status: `completed`

This package does not create independent physical-validity ground truth. It records exactly which information is shared across training-target construction, the primary OBB verifier, and the point/mesh audit, then hash-verifies the compact analyses used to probe that dependence.

## Verified evidence

- Linear point/mesh agreement: 14/15 cells have a negative Violation change.
- MLP point/mesh agreement: 14/15 cells have a negative Violation change.
- Uncertainty-policy check: primary/decidable/pessimistic Violation is non-increasing in 30/30, 30/30, and 30/30 cells, respectively.
- Feature-removal, counterfactual-sensitivity, and component-removal artifacts are present and their checksums match.

The dependency matrix and complete evidence index are stored in the adjacent CSV files.
