#!/usr/bin/env python3
"""Regenerate main-paper Tables 1--3 and Figure 3 from derived rows."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


SOURCES = ("open3dsg", "vlsat", "sgfn")
SOURCE_LABELS = {
    "open3dsg": "Open3DSG",
    "vlsat": "VL-SAT",
    "sgfn": "SGFN",
}
KS = (5, 10, 20, 50, 100)

TABLE1_METHODS = (
    ("Source", "rank_source"),
    ("RelCompat3D-Linear", "rank_linear"),
    ("RelCompat3D-MLP", "rank_mlp"),
    ("RankAvg", "rank_rankavg"),
    ("RRF", "rank_rrf"),
    ("Product (all families)", "rank_product_all_families"),
)
TABLE2_METHODS = (
    ("RelCompat3D-Linear", "rank_linear"),
    ("RelCompat3D-MLP", "rank_mlp"),
    ("Wrong predicate", "rank_wrong_predicate"),
    ("Wrong pair", "rank_wrong_pair"),
    ("Shuffled geometry", "rank_shuffled_geometry"),
    ("Fixed-predicate swap", "rank_fixed_predicate_swap"),
    ("Distance only", "rank_distance_only"),
    ("Compatibility only", "rank_compatibility_only"),
)
DECIDED = {"satisfied", "violated"}
MEASURED = {"satisfied", "uncertain", "violated"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_gz(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def read_reference_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def empty_metric_cells(
    methods: tuple[tuple[str, str], ...],
    ks: tuple[int, ...],
) -> dict[str, dict[int, dict[str, int]]]:
    return {
        label: {
            k: {
                "recall_num": 0,
                "violated": 0,
                "status_den": 0,
                "selected": 0,
            }
            for k in ks
        }
        for label, _ in methods
    }


def aggregate_rows(
    rows_dir: Path,
    ground_truth_denominator: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    table1 = {
        source: empty_metric_cells(TABLE1_METHODS, KS) for source in SOURCES
    }
    table2 = {
        source: empty_metric_cells(TABLE2_METHODS, (50, 100))
        for source in SOURCES
    }
    table3 = {
        source: {
            method: {
                "selected_scope": 0,
                "satisfied": 0,
                "uncertain": 0,
                "violated": 0,
                "unsupported": 0,
            }
            for method in ("Source", "RelCompat3D-Linear")
        }
        for source in SOURCES
    }
    audits: dict[str, Any] = {}

    for source in SOURCES:
        path = rows_dir / f"{source}_candidates.csv.gz"
        seen_rows: set[str] = set()
        rank_seen: dict[str, set[int]] = {
            rank: set()
            for _, rank in (*TABLE1_METHODS, *TABLE2_METHODS)
        }
        candidate_count = exact_pool = 0
        for row in read_csv_gz(path):
            candidate_count += 1
            row_uid = row["row_uid"]
            if row_uid in seen_rows:
                raise ValueError(f"duplicate_row_uid:{source}:{row_uid}")
            seen_rows.add(row_uid)
            exact = int(row["exact_match"])
            exact_pool += exact
            status = row["verifier_status"]

            for label, rank_column in TABLE1_METHODS:
                rank = int(row[rank_column])
                if rank in rank_seen[rank_column]:
                    pass
                rank_seen[rank_column].add(rank)
                for k in KS:
                    if rank > k:
                        continue
                    cell = table1[source][label][k]
                    cell["selected"] += 1
                    cell["recall_num"] += exact
                    if status in MEASURED:
                        cell["status_den"] += 1
                        cell["violated"] += int(status == "violated")

            for label, rank_column in TABLE2_METHODS:
                rank = int(row[rank_column])
                for k in (50, 100):
                    if rank > k:
                        continue
                    cell = table2[source][label][k]
                    cell["selected"] += 1
                    cell["recall_num"] += exact
                    if status in MEASURED:
                        cell["status_den"] += 1
                        cell["violated"] += int(status == "violated")

            if row["family"] in {"proximity", "relative_vertical"}:
                agreement = row["surface_agreement_status"] or "unsupported"
                for label, rank_column in (
                    ("Source", "rank_source"),
                    ("RelCompat3D-Linear", "rank_linear"),
                ):
                    if int(row[rank_column]) > 50:
                        continue
                    cell = table3[source][label]
                    cell["selected_scope"] += 1
                    if agreement in {
                        "satisfied",
                        "uncertain",
                        "violated",
                        "unsupported",
                    }:
                        cell[agreement] += 1
                    else:
                        cell["unsupported"] += 1
        audits[source] = {
            "candidate_rows": candidate_count,
            "pool_exact_match_rows": exact_pool,
            "unique_row_uids": len(seen_rows),
        }

    for source in SOURCES:
        for method in table1[source].values():
            for cell in method.values():
                cell["recall"] = ratio(
                    cell["recall_num"], ground_truth_denominator
                )
                cell["violation"] = ratio(
                    cell["violated"], cell["status_den"]
                )
        for method in table2[source].values():
            for cell in method.values():
                cell["recall"] = ratio(
                    cell["recall_num"], ground_truth_denominator
                )
                cell["violation"] = ratio(
                    cell["violated"], cell["status_den"]
                )
        for method in table3[source].values():
            measured = (
                method["satisfied"] + method["uncertain"] + method["violated"]
            )
            decidable = method["satisfied"] + method["violated"]
            method["violation"] = ratio(method["violated"], measured)
            method["measured_coverage"] = ratio(
                measured, method["selected_scope"]
            )
            method["decidable_coverage"] = ratio(
                decidable, method["selected_scope"]
            )
    return table1, table2, table3, audits


def table1_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        for label, _ in TABLE1_METHODS:
            row: dict[str, Any] = {
                "predictor": SOURCE_LABELS[source],
                "ranking_rule": label,
            }
            for k in KS:
                cell = summary[source][label][k]
                row[f"R@{k}"] = 100.0 * cell["recall"]
                row[f"V@{k}"] = 100.0 * cell["violation"]
            rows.append(row)
    return rows


def table2_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in SOURCES:
        for label, _ in TABLE2_METHODS:
            rows.append(
                {
                    "predictor": SOURCE_LABELS[source],
                    "condition": label,
                    "R@50": 100.0 * summary[source][label][50]["recall"],
                    "V@50": 100.0 * summary[source][label][50]["violation"],
                    "R@100": 100.0 * summary[source][label][100]["recall"],
                    "V@100": 100.0 * summary[source][label][100]["violation"],
                }
            )
    return rows


def table3_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in ("vlsat", "open3dsg", "sgfn"):
        source_cell = summary[source]["Source"]
        linear_cell = summary[source]["RelCompat3D-Linear"]
        rows.append(
            {
                "predictor": SOURCE_LABELS[source],
                "source_violation": 100.0 * source_cell["violation"],
                "linear_violation": 100.0 * linear_cell["violation"],
                "delta_violation_pp": 100.0
                * (linear_cell["violation"] - source_cell["violation"]),
                "measured_coverage": 100.0 * linear_cell["measured_coverage"],
                "decidable_coverage": 100.0 * linear_cell["decidable_coverage"],
            }
        )
    return rows


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_table_tex(path: Path, rows: list[dict[str, Any]], table: int) -> None:
    lines: list[str] = []
    if table == 1:
        lines.extend(
            [
                r"\begin{tabular}{llrrrrrrrrrr}",
                r"\toprule",
                r"Predictor & Ranking rule & R@5 & V@5 & R@10 & V@10 & R@20 & V@20 & R@50 & V@50 & R@100 & V@100 \\",
                r"\midrule",
            ]
        )
        for row in rows:
            values = [
                tex_escape(str(row["predictor"])),
                tex_escape(str(row["ranking_rule"])),
                *[
                    f"{float(row[f'{metric}@{k}']):.2f}"
                    for k in KS
                    for metric in ("R", "V")
                ],
            ]
            lines.append(" & ".join(values) + r" \\")
    elif table == 2:
        lines.extend(
            [
                r"\begin{tabular}{llrrrr}",
                r"\toprule",
                r"Predictor & Condition & R@50 & V@50 & R@100 & V@100 \\",
                r"\midrule",
            ]
        )
        for row in rows:
            values = [
                tex_escape(str(row["predictor"])),
                tex_escape(str(row["condition"])),
                *[
                    f"{float(row[name]):.2f}"
                    for name in ("R@50", "V@50", "R@100", "V@100")
                ],
            ]
            lines.append(" & ".join(values) + r" \\")
    else:
        lines.extend(
            [
                r"\begin{tabular}{lrrrrr}",
                r"\toprule",
                r"Predictor & Source & Linear & $\Delta V$ & Measured & Decidable \\",
                r"\midrule",
            ]
        )
        for row in rows:
            values = [
                tex_escape(str(row["predictor"])),
                *[
                    f"{float(row[name]):.2f}"
                    for name in (
                        "source_violation",
                        "linear_violation",
                        "delta_violation_pp",
                        "measured_coverage",
                        "decidable_coverage",
                    )
                ],
            ]
            lines.append(" & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def expected_table1(protocol: dict[str, Any], root: Path) -> dict[tuple[str, str, int, str], float]:
    refs = protocol["canonical_references"]
    routed = read_reference_csv(resolve(root, refs["table1_routed"]["path"]))
    method_map = {
        "source_score": "Source",
        "routed_product": "RelCompat3D-Linear",
        "routed_matched_mlp": "RelCompat3D-MLP",
        "routed_rank_average": "RankAvg",
        "routed_rrf": "RRF",
    }
    expected: dict[tuple[str, str, int, str], float] = {}
    for row in routed:
        label = method_map[row["method"]]
        for metric in ("recall", "violation"):
            expected[(row["source"], label, int(row["k"]), metric)] = float(
                row[metric]
            )

    common = read_reference_csv(
        resolve(root, refs["table1_all_family_vlsat_sgfn"]["path"])
    )
    for row in common:
        if (
            row["source"] in {"vlsat", "sgfn"}
            and row["method"] == "structured_product"
        ):
            expected[
                (row["source"], "Product (all families)", int(row["k"]), "recall")
            ] = float(row["recall"])
            expected[
                (
                    row["source"],
                    "Product (all families)",
                    int(row["k"]),
                    "violation",
                )
            ] = float(row["violation"])
    open_rows = read_reference_csv(
        resolve(root, refs["table1_all_family_open3dsg"]["path"])
    )
    for row in open_rows:
        if (
            row["route"] == "official_strict_full_548"
            and row["method"] == "structured_product"
        ):
            expected[
                ("open3dsg", "Product (all families)", int(row["k"]), "recall")
            ] = float(row["recall"])
            expected[
                (
                    "open3dsg",
                    "Product (all families)",
                    int(row["k"]),
                    "violation",
                )
            ] = float(row["violation"])
    return expected


def expected_table2(protocol: dict[str, Any], root: Path) -> dict[tuple[str, str, int, str], float]:
    refs = protocol["canonical_references"]
    rows = read_reference_csv(
        resolve(root, refs["table2_linear_controls"]["path"])
    )
    label_map = {
        "structured_product": "RelCompat3D-Linear",
        "wrong_predicate_product": "Wrong predicate",
        "wrong_pair_product": "Wrong pair",
        "shuffled_geometry_product": "Shuffled geometry",
        "endpoint_swap_fixed_label_product": "Fixed-predicate swap",
        "distance_only": "Distance only",
        "compatibility_only": "Compatibility only",
    }
    expected: dict[tuple[str, str, int, str], float] = {}
    for row in rows:
        if row["method"] not in label_map:
            continue
        label = label_map[row["method"]]
        for metric in ("recall", "violation"):
            expected[(row["source"], label, int(row["k"]), metric)] = float(
                row[metric]
            )
    routed = read_reference_csv(resolve(root, refs["table1_routed"]["path"]))
    for row in routed:
        if row["method"] == "routed_matched_mlp" and int(row["k"]) in {50, 100}:
            for metric in ("recall", "violation"):
                expected[
                    (
                        row["source"],
                        "RelCompat3D-MLP",
                        int(row["k"]),
                        metric,
                    )
                ] = float(row[metric])
    return expected


def expected_table3(protocol: dict[str, Any], root: Path) -> dict[tuple[str, str], float]:
    path = resolve(
        root, protocol["canonical_references"]["table3_surface"]["path"]
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[tuple[str, str], float] = {}
    for source in ("vlsat", "open3dsg", "sgfn"):
        cells = summary["results"][source]["audits"]["consensus"]
        source_cell = cells["source"]["50"]
        linear_cell = cells["relcompat3d"]["50"]
        expected[(source, "source_violation")] = source_cell["violation"]["point"]
        expected[(source, "linear_violation")] = linear_cell["violation"]["point"]
        expected[(source, "delta_violation")] = (
            linear_cell["violation"]["point"]
            - source_cell["violation"]["point"]
        )
        expected[(source, "measured_coverage")] = linear_cell["coverage"]["point"]
        expected[(source, "decidable_coverage")] = linear_cell[
            "decidable_coverage"
        ]["point"]
    return expected


def canonical_validation(
    table1: dict[str, Any],
    table2: dict[str, Any],
    table3: dict[str, Any],
    protocol: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tolerance = float(protocol["scope"]["numerical_tolerance"])
    for key, expected in expected_table1(protocol, root).items():
        source, method, k, metric = key
        actual = float(table1[source][method][k][metric])
        error = abs(actual - expected)
        rows.append(
            {
                "table": "Table 1",
                "predictor": source,
                "method_or_field": method,
                "k": k,
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "abs_error": error,
                "passed": error <= tolerance,
            }
        )
    for key, expected in expected_table2(protocol, root).items():
        source, method, k, metric = key
        actual = float(table2[source][method][k][metric])
        error = abs(actual - expected)
        rows.append(
            {
                "table": "Table 2",
                "predictor": source,
                "method_or_field": method,
                "k": k,
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "abs_error": error,
                "passed": error <= tolerance,
            }
        )
    for key, expected in expected_table3(protocol, root).items():
        source, field = key
        if field == "source_violation":
            actual = table3[source]["Source"]["violation"]
        elif field == "linear_violation":
            actual = table3[source]["RelCompat3D-Linear"]["violation"]
        elif field == "delta_violation":
            actual = (
                table3[source]["RelCompat3D-Linear"]["violation"]
                - table3[source]["Source"]["violation"]
            )
        else:
            actual = table3[source]["RelCompat3D-Linear"][field]
        error = abs(float(actual) - expected)
        rows.append(
            {
                "table": "Table 3",
                "predictor": source,
                "method_or_field": field,
                "k": 50,
                "metric": field,
                "expected": expected,
                "actual": actual,
                "abs_error": error,
                "passed": error <= tolerance,
            }
        )
    return rows


def figure_data_rows(table1: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in ("vlsat", "open3dsg", "sgfn"):
        for method in ("Source", "RelCompat3D-Linear", "RelCompat3D-MLP"):
            for k in KS:
                cell = table1[source][method][k]
                rows.append(
                    {
                        "predictor": SOURCE_LABELS[source],
                        "method": method,
                        "k": k,
                        "recall_percent": 100.0 * cell["recall"],
                        "violation_percent": 100.0 * cell["violation"],
                    }
                )
    return rows


def render_svg(path: Path, data: list[dict[str, Any]]) -> None:
    width, height = 1500, 580
    margin_x, plot_y, plot_h = 80, 100, 390
    panel_w, gap = 400, 70
    colors = {
        "Source": "#707070",
        "RelCompat3D-Linear": "#008B8B",
        "RelCompat3D-MLP": "#6F42C1",
    }
    markers = {
        "Source": "circle",
        "RelCompat3D-Linear": "square",
        "RelCompat3D-MLP": "triangle",
    }
    panels = ("VL-SAT", "Open3DSG", "SGFN")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111}.axis{stroke:#111;stroke-width:2}.grid{stroke:#ddd;stroke-width:1}.line{fill:none;stroke-width:3}</style>',
    ]
    for panel_index, predictor in enumerate(panels):
        x0 = margin_x + panel_index * (panel_w + gap)
        cells = [row for row in data if row["predictor"] == predictor]
        recalls = [float(row["recall_percent"]) for row in cells]
        violations = [float(row["violation_percent"]) for row in cells]
        xmin, xmax = min(recalls), max(recalls)
        ymin, ymax = 0.0, max(violations) * 1.12 + 1e-9
        xpad = max((xmax - xmin) * 0.08, 1.0)
        xmin -= xpad
        xmax += xpad

        def px(value: float) -> float:
            return x0 + (value - xmin) / (xmax - xmin) * panel_w

        def py(value: float) -> float:
            return plot_y + plot_h - (value - ymin) / (ymax - ymin) * plot_h

        parts.append(
            f'<text x="{x0}" y="55" font-size="26" font-weight="700">({chr(97 + panel_index)}) {html.escape(predictor)}</text>'
        )
        parts.append(
            f'<line class="axis" x1="{x0}" y1="{plot_y + plot_h}" x2="{x0 + panel_w}" y2="{plot_y + plot_h}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{x0}" y1="{plot_y}" x2="{x0}" y2="{plot_y + plot_h}"/>'
        )
        parts.append(
            f'<text x="{x0 + panel_w / 2}" y="{height - 22}" font-size="22" text-anchor="middle">Recall@K (%)</text>'
        )
        if panel_index == 0:
            parts.append(
                f'<text x="25" y="{plot_y + plot_h / 2}" font-size="22" text-anchor="middle" transform="rotate(-90 25 {plot_y + plot_h / 2})">Violation@K (%)</text>'
            )
        for method in colors:
            method_rows = sorted(
                (row for row in cells if row["method"] == method),
                key=lambda row: int(row["k"]),
            )
            points = " ".join(
                f"{px(float(row['recall_percent'])):.2f},{py(float(row['violation_percent'])):.2f}"
                for row in method_rows
            )
            dash = ' stroke-dasharray="10 8"' if method == "Source" else ""
            parts.append(
                f'<polyline class="line" points="{points}" stroke="{colors[method]}"{dash}/>'
            )
            for row in method_rows:
                x = px(float(row["recall_percent"]))
                y = py(float(row["violation_percent"]))
                if markers[method] == "circle":
                    parts.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{colors[method]}"/>'
                    )
                elif markers[method] == "square":
                    parts.append(
                        f'<rect x="{x - 6:.2f}" y="{y - 6:.2f}" width="12" height="12" fill="{colors[method]}"/>'
                    )
                else:
                    parts.append(
                        f'<path d="M{x:.2f},{y - 7:.2f} L{x - 7:.2f},{y + 6:.2f} L{x + 7:.2f},{y + 6:.2f} Z" fill="{colors[method]}"/>'
                    )
                if method == "Source":
                    parts.append(
                        f'<text x="{x + 7:.2f}" y="{y - 8:.2f}" font-size="15">{int(row["k"])}</text>'
                    )
    legend_x, legend_y = 1110, 32
    for index, method in enumerate(colors):
        y = legend_y + index * 25
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 36}" y2="{y}" stroke="{colors[method]}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 45}" y="{y + 7}" font-size="18">{html.escape(method)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_raster(path_png: Path, path_pdf: Path, data: list[dict[str, Any]]) -> None:
    width, height = 1800, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    title_font = ImageFont.load_default(size=28)
    colors = {
        "Source": (112, 112, 112),
        "RelCompat3D-Linear": (0, 139, 139),
        "RelCompat3D-MLP": (111, 66, 193),
    }
    margin_x, plot_y, plot_h = 90, 120, 470
    panel_w, gap = 470, 90
    for panel_index, predictor in enumerate(("VL-SAT", "Open3DSG", "SGFN")):
        x0 = margin_x + panel_index * (panel_w + gap)
        cells = [row for row in data if row["predictor"] == predictor]
        recalls = [float(row["recall_percent"]) for row in cells]
        violations = [float(row["violation_percent"]) for row in cells]
        xmin, xmax = min(recalls), max(recalls)
        xpad = max((xmax - xmin) * 0.08, 1.0)
        xmin, xmax = xmin - xpad, xmax + xpad
        ymax = max(violations) * 1.12 + 1e-9

        def px(value: float) -> float:
            return x0 + (value - xmin) / (xmax - xmin) * panel_w

        def py(value: float) -> float:
            return plot_y + plot_h - value / ymax * plot_h

        draw.text(
            (x0, 55),
            f"({chr(97 + panel_index)}) {predictor}",
            fill="black",
            font=title_font,
        )
        draw.line((x0, plot_y, x0, plot_y + plot_h), fill="black", width=3)
        draw.line(
            (x0, plot_y + plot_h, x0 + panel_w, plot_y + plot_h),
            fill="black",
            width=3,
        )
        draw.text(
            (x0 + panel_w // 2 - 80, height - 60),
            "Recall@K (%)",
            fill="black",
            font=font,
        )
        for method in colors:
            points = [
                (
                    px(float(row["recall_percent"])),
                    py(float(row["violation_percent"])),
                )
                for row in sorted(
                    (row for row in cells if row["method"] == method),
                    key=lambda row: int(row["k"]),
                )
            ]
            draw.line(points, fill=colors[method], width=5)
            for x, y in points:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[method])
    image.save(path_png, dpi=(300, 300))
    image.save(path_pdf, "PDF", resolution=300.0)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    protocol_path = resolve(root, args.protocol)
    rows_dir = resolve(root, args.rows)
    out = resolve(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"nonempty_output:{out}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_table_row_export":
        raise ValueError("protocol_not_frozen")

    manifest_path = rows_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("table_rows_incomplete")
    for name, spec in manifest["files"].items():
        path = rows_dir / name
        if not path.exists() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"table_rows_hash_mismatch:{name}")
    for name, spec in protocol["canonical_references"].items():
        path = resolve(root, spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"canonical_reference_hash_mismatch:{name}")

    ground_truth_rows = sum(
        1 for _ in read_csv_gz(rows_dir / "ground_truth.csv.gz")
    )
    table1, table2, table3, row_audits = aggregate_rows(
        rows_dir, ground_truth_rows
    )
    table1_csv = table1_rows(table1)
    table2_csv = table2_rows(table2)
    table3_csv = table3_rows(table3)
    validation_rows = canonical_validation(
        table1, table2, table3, protocol, root
    )
    figure_rows = figure_data_rows(table1)
    validations = {
        "table_rows_hashes_match": True,
        "canonical_reference_hashes_match": True,
        "ground_truth_denominator_3972": (
            ground_truth_rows
            == protocol["scope"]["expected_ground_truth_rows"]
        ),
        "candidate_row_counts_match": {
            source: row_audits[source]["candidate_rows"]
            for source in SOURCES
        }
        == protocol["scope"]["expected_candidate_rows"],
        "all_canonical_cells_within_tolerance": all(
            str(row["passed"]).lower() == "true" for row in validation_rows
        ),
        "all_table1_selected_counts_bounded": all(
            cell["selected"] <= k * protocol["scope"]["expected_contexts"]
            for source in table1.values()
            for method in source.values()
            for k, cell in method.items()
        ),
        "table3_all_selected_rows_have_accounted_status": all(
            cell["selected_scope"]
            == cell["satisfied"]
            + cell["uncertain"]
            + cell["violated"]
            + cell["unsupported"]
            for source in table3.values()
            for cell in source.values()
        ),
    }
    status = "completed" if all(
        value if isinstance(value, bool) else bool(value)
        for value in validations.values()
    ) else "failed_validation"

    out.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "table1.csv": out / "table1.csv",
        "table2.csv": out / "table2.csv",
        "table3.csv": out / "table3.csv",
        "canonical_validation.csv": out / "canonical_validation.csv",
        "figure3_data.csv": out / "figure3_data.csv",
        "table1.tex": out / "table1.tex",
        "table2.tex": out / "table2.tex",
        "table3.tex": out / "table3.tex",
        "Figure3.svg": out / "Figure3.svg",
        "Figure3.png": out / "Figure3.png",
        "Figure3.pdf": out / "Figure3.pdf",
    }
    write_csv(output_paths["table1.csv"], table1_csv)
    write_csv(output_paths["table2.csv"], table2_csv)
    write_csv(output_paths["table3.csv"], table3_csv)
    write_csv(output_paths["canonical_validation.csv"], validation_rows)
    write_csv(output_paths["figure3_data.csv"], figure_rows)
    write_table_tex(output_paths["table1.tex"], table1_csv, 1)
    write_table_tex(output_paths["table2.tex"], table2_csv, 2)
    write_table_tex(output_paths["table3.tex"], table3_csv, 3)
    render_svg(output_paths["Figure3.svg"], figure_rows)
    render_raster(
        output_paths["Figure3.png"],
        output_paths["Figure3.pdf"],
        figure_rows,
    )

    max_error = max(float(row["abs_error"]) for row in validation_rows)
    summary = {
        "schema_version": "relcompat3d_row_reproduction_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "table_rows_manifest_sha256": sha256_file(manifest_path),
        "ground_truth_rows": ground_truth_rows,
        "candidate_row_audits": row_audits,
        "canonical_cells": len(validation_rows),
        "maximum_absolute_error": max_error,
        "tolerance": protocol["scope"]["numerical_tolerance"],
        "validations": validations,
        "claim_boundary": protocol["claim_boundary"],
        "docker_command": (
            "env UID=$(id -u) GID=$(id -g) docker compose "
            "-f configs/relcompat3d/compose.yaml run --rm "
            "relcompat3d_reproduce_rows"
        ),
    }
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    summary_md = out / "summary.md"
    summary_md.write_text(
        "\n".join(
            (
                "# Row-Level Paper Reproduction",
                "",
                f"Status: `{status}`",
                "",
                f"- Candidate rows: {sum(cell['candidate_rows'] for cell in row_audits.values()):,}",
                f"- Ground-truth rows: {ground_truth_rows:,}",
                f"- Canonical cells checked: {len(validation_rows)}",
                f"- Maximum absolute error: {max_error:.3e}",
                f"- Required tolerance: {float(protocol['scope']['numerical_tolerance']):.1e}",
                "",
                "The regenerated Tables 1--3 and Figure 3 use only the local table rows. The rendering verifies the numerical figure data.",
                "",
            )
        ),
        encoding="utf-8",
    )
    output_paths["summary.json"] = summary_path
    output_paths["summary.md"] = summary_md
    write_json(
        out / "manifest.json",
        {
            "schema_version": "relcompat3d_row_reproduction_manifest_v1",
            "status": status,
            "protocol": {
                "path": relpath(root, protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "table_rows": {
                "path": relpath(root, rows_dir),
                "manifest_sha256": sha256_file(manifest_path),
            },
            "outputs": {
                name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for name, path in output_paths.items()
            },
            "validations": validations,
            "maximum_absolute_error": max_error,
            "docker_command": summary["docker_command"],
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "canonical_cells": len(validation_rows),
                "maximum_absolute_error": max_error,
                "validations": validations,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
