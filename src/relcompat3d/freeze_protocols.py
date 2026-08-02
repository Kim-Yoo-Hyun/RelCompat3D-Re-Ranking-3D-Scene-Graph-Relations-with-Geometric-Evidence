#!/usr/bin/env python3
"""Freeze post-fit RelCompat3D protocols around the no-family-indicator model lock."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_REL = Path("experiments/RelCompat3D_geom_reliability/main_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("initial", "downstream"), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"protocol_already_frozen:{path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_completed(path: Path) -> dict[str, Any]:
    payload = load(path)
    if payload.get("status") != "completed":
        raise ValueError(f"input_not_completed:{path}:{payload.get('status')}")
    return payload


def freeze_initial(root: Path, exp: Path, protocols: Path) -> list[Path]:
    fit = exp / "fit"
    lock = load(fit / "final_lock.json")
    if lock.get("status") != "locked_before_official_validation":
        raise ValueError("fit_model_not_locked")
    structured_rel = str(ROOT_REL / "fit/structured_models.json")
    strict_rel = str(ROOT_REL / "fit/strict_models.json")
    manifest_rel = str(ROOT_REL / "fit/manifest.json")
    diagnostics_rel = str(ROOT_REL / "fit/internal_dev_diagnostics.json")
    if sha256(fit / "structured_models.json") != lock["structured_model_sha256"]:
        raise ValueError("structured_hash_mismatch")
    if sha256(fit / "strict_models.json") != lock["strict_model_sha256"]:
        raise ValueError("strict_hash_mismatch")

    structured = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/structured_main_v1/protocol.json")
    )
    structured["created_at_kst"] = "2026-07-20"
    structured["classification"] = "main_experiment_fixed_model_benchmark_evaluation"
    structured["main_compatibility"]["artifact_id"] = "main_experiment_orbit_pairwise_projected"
    structured["main_compatibility"]["definition"] = (
        "family-specific logistic compatibility without a family-indicator input, trained with the unchanged "
        "BCE, relation-preserving augmentation, and linked-counterfactual margin; inference transformation "
        "averaging and family-aware score use are unchanged"
    )
    structured["main_compatibility"]["model_bundle_sha256"] = lock["structured_model_sha256"]
    structured["inputs"].update(
        {
            "structured_models": structured_rel,
            "strict_models": strict_rel,
            "relation_manifest": manifest_rel,
            "relation_diagnostics": diagnostics_rel,
        }
    )
    structured["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    routing = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/support_contact_routing_v1/protocol.json")
    )
    routing["created_at_kst"] = "2026-07-20"
    routing["purpose"] = "Re-evaluate the locked no-family-indicator compatibility under the unchanged family-aware ranking rule."
    routing["inputs"]["structured_models"] = structured_rel
    routing["inputs"]["strict_models"] = strict_rel
    routing["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    open_route = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/open3dsg_official_route_v1/protocol.json")
    )
    open_route["created_at_kst"] = "2026-07-20"
    open_route["purpose"] = "Re-evaluate Open3DSG coverage routes with the locked no-family-indicator compatibility."
    open_route["inputs"]["structured_models"] = structured_rel
    open_route["inputs"]["strict_models"] = strict_rel
    open_route["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    nonlinear = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/supervision_matched_nonlinear_v1/protocol.json")
    )
    nonlinear["created_at_kst"] = "2026-07-20"
    nonlinear["purpose"] = (
        "Regenerate the unchanged shared nonlinear baseline and compare it with the locked "
        "no-family-indicator linear compatibility."
    )
    nonlinear["inputs"]["structured_models"] = structured_rel
    nonlinear["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    outputs = [
        protocols / "structured_main.json",
        protocols / "support_routing.json",
        protocols / "open3dsg_route.json",
        protocols / "nonlinear.json",
    ]
    for path, payload in zip(outputs, (structured, routing, open_route, nonlinear)):
        write(path, payload)
    freeze = {
        "schema_version": "relcompat3d_main_experiment_initial_protocol_freeze_v1",
        "status": "frozen_before_official_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock": lock,
        "protocols": {path.name: sha256(path) for path in outputs},
        "selection_or_tuning_allowed": False,
    }
    freeze_path = protocols / "initial_freeze.json"
    write(freeze_path, freeze)
    return [*outputs, freeze_path]


def freeze_downstream(root: Path, exp: Path, protocols: Path) -> list[Path]:
    fit = exp / "fit"
    lock = load(fit / "final_lock.json")
    evaluation = exp / "evaluation"
    require_completed(evaluation / "structured_main/manifest.json")
    require_completed(evaluation / "support_routing/manifest.json")
    require_completed(evaluation / "open3dsg_route/manifest.json")
    require_completed(evaluation / "nonlinear/manifest.json")

    structured_rel = str(ROOT_REL / "fit/structured_models.json")
    strict_rel = str(ROOT_REL / "fit/strict_models.json")
    nonlinear_rel = str(ROOT_REL / "evaluation/nonlinear/models.json")
    routing_summary_rel = str(ROOT_REL / "evaluation/support_routing/summary.json")
    open_summary_rel = str(ROOT_REL / "evaluation/open3dsg_route/summary.json")
    nonlinear_hash = sha256(evaluation / "nonlinear/models.json")
    routing_hash = sha256(evaluation / "support_routing/summary.json")
    open_hash = sha256(evaluation / "open3dsg_route/summary.json")

    comparator = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/routed_comparators_v1/protocol.json")
    )
    comparator["created_at_kst"] = "2026-07-20"
    comparator["purpose"] = "All-K matched comparison for the locked no-family-indicator compatibility."
    comparator["inputs"].update(
        {
            "structured_models": structured_rel,
            "strict_models": strict_rel,
            "nonlinear_models": nonlinear_rel,
            "routing_reference": routing_summary_rel,
            "open3dsg_reference": open_summary_rel,
        }
    )
    comparator["locked_sha256"] = {
        "structured_models": lock["structured_model_sha256"],
        "strict_models": lock["strict_model_sha256"],
        "nonlinear_models": nonlinear_hash,
    }
    comparator["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    ablation = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/structured_ablation_v1/routed_public_full_protocol.json")
    )
    ablation["created_at_kst"] = "2026-07-20"
    ablation["classification"] = "fixed no-family-indicator model ablation on the public/full 548-context target"
    ablation["method"]["model_sha256"] = lock["structured_model_sha256"]
    ablation["method"]["fit"] = "main_experiment strict 1061-scan train-only refit"
    ablation["inputs"]["structured_models"] = {
        "path": structured_rel,
        "sha256": lock["structured_model_sha256"],
    }
    ablation["inputs"]["strict_models"] = {
        "path": strict_rel,
        "sha256": lock["strict_model_sha256"],
    }
    ablation["inputs"]["routing_summary"] = {
        "path": routing_summary_rel,
        "sha256": routing_hash,
    }
    ablation["inputs"]["open3dsg_official_summary"] = {
        "path": open_summary_rel,
        "sha256": open_hash,
    }
    ablation["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    scan = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/support_contact_routing_v1/scan_cluster_protocol.json")
    )
    scan["created_at_kst"] = "2026-07-20"
    scan["routing_protocol"] = str(ROOT_REL / "protocols/support_routing.json")
    scan["routing_summary"] = routing_summary_rel
    scan["open3dsg_official_summary"] = open_summary_rel
    scan["model_sha256"] = lock["structured_model_sha256"]
    scan["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    structured_scan = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/structured_main_v1/scan_cluster_protocol.json")
    )
    structured_scan["created_at_kst"] = "2026-07-20"
    structured_scan["main_protocol"] = str(ROOT_REL / "protocols/structured_main.json")
    structured_scan["current_summary"] = str(ROOT_REL / "evaluation/structured_main/summary.json")
    structured_scan["model_sha256"] = lock["structured_model_sha256"]
    structured_scan["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    surface = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/orthogonal_geometry_audit_v1/protocol.json")
    )
    surface["created_at_kst"] = "2026-07-20"
    surface["purpose"] = (
        "Re-evaluate the locked no-family-indicator ranking with the unchanged "
        "point-, mesh-, and consensus-surface audit."
    )
    surface["inputs"]["structured_models"] = structured_rel
    surface["inputs"]["strict_models"] = strict_rel
    surface["locked_sha256"] = {
        "structured_models": lock["structured_model_sha256"],
        "strict_models": lock["strict_model_sha256"],
    }
    surface["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    heldout = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/held_out_primitive_v1/protocol.json")
    )
    heldout["created_at_kst"] = "2026-07-20"
    heldout["purpose"] = (
        "Repeat the unchanged feature-removal analysis around the locked "
        "no-family-indicator compatibility model."
    )
    heldout["inputs"].update(
        {
            "strict_models": strict_rel,
            "main_models": structured_rel,
            "routing_summary": routing_summary_rel,
            "open3dsg_official_summary": open_summary_rel,
        }
    )
    heldout["locked_hashes"] = {
        "main_models_sha256": lock["structured_model_sha256"],
        "strict_models_sha256": lock["strict_model_sha256"],
    }
    heldout["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    counterfactual = copy.deepcopy(
        load(root / "experiments/RelCompat3D_geom_reliability/counterfactual_sensitivity_v1/protocol.json")
    )
    counterfactual["created_at_kst"] = "2026-07-20"
    counterfactual["purpose"] = (
        "Repeat the unchanged one-factor counterfactual sensitivity while "
        "removing the constant family indicator from every family-specific head."
    )
    counterfactual["parameterization"] = {
        "id": "main_experiment",
        "remove_constant_family_indicator": True,
        "other_changes": "none",
    }
    counterfactual["inputs"].update(
        {
            "main_models": structured_rel,
            "routing_summary": routing_summary_rel,
            "open3dsg_summary": open_summary_rel,
        }
    )
    counterfactual["locked_sha256"] = {
        "main_models": lock["structured_model_sha256"]
    }
    counterfactual["parent_lock"] = str(ROOT_REL / "fit/final_lock.json")

    outputs = [
        protocols / "routed_comparators.json",
        protocols / "routed_ablation.json",
        protocols / "scan_cluster.json",
        protocols / "structured_scan_cluster.json",
        protocols / "surface_audit.json",
        protocols / "held_out_primitive.json",
        protocols / "counterfactual_sensitivity.json",
    ]
    for path, payload in zip(
        outputs,
        (
            comparator,
            ablation,
            scan,
            structured_scan,
            surface,
            heldout,
            counterfactual,
        ),
    ):
        write(path, payload)
    freeze = {
        "schema_version": "relcompat3d_main_experiment_downstream_protocol_freeze_v1",
        "status": "frozen_before_downstream_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_lock": lock,
        "upstream_hashes": {
            "nonlinear_models": nonlinear_hash,
            "routing_summary": routing_hash,
            "open3dsg_summary": open_hash,
        },
        "protocols": {path.name: sha256(path) for path in outputs},
        "selection_or_tuning_allowed": False,
    }
    freeze_path = protocols / "downstream_freeze.json"
    write(freeze_path, freeze)
    return [*outputs, freeze_path]


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    exp = root / ROOT_REL
    protocols = exp / "protocols"
    outputs = (
        freeze_initial(root, exp, protocols)
        if args.phase == "initial"
        else freeze_downstream(root, exp, protocols)
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "phase": args.phase,
                "outputs": [str(path.relative_to(root)) for path in outputs],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
