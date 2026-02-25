#!/usr/bin/env python3
"""Generate comprehensive metadata statistics and visualizations.

Reads annotated JSONL files and produces PNG plots showing jersey color,
number color, and border distributions -- both overall and per-class.

Usage::

    python scripts/metadata_stats.py --data-root /path/to/dataset
    python scripts/metadata_stats.py --data-root /path/to/dataset \
        --split train --output-dir outputs/metadata_stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add project root to path so we can import classifier_training
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier_training.data.utils import get_files  # noqa: E402

COLOR_HEX = {
    "white": "#F5F5F5",
    "black": "#222222",
    "red": "#E53935",
    "blue": "#1E88E5",
    "navy": "#1A237E",
    "yellow": "#FDD835",
    "green": "#43A047",
    "purple": "#8E24AA",
    "orange": "#FB8C00",
    "grey": "#9E9E9E",
    "maroon": "#880E4F",
    "teal": "#00897B",
    "pink": "#EC407A",
    "unknown": "#BDBDBD",
}


def load_records(
    split_dir: Path,
) -> list[dict[str, object]]:
    """Load all JSONL records from annotation files."""
    records: list[dict[str, object]] = []
    ann_files = get_files(split_dir, (".jsonl",))
    for ann_path in ann_files:
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def collect_stats(
    records: list[dict[str, object]],
) -> tuple[
    Counter[str],
    Counter[str],
    Counter[bool],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    dict[str, Counter[bool]],
    int,
    int,
    dict[str, tuple[int, int]],
]:
    """Collect metadata statistics from records.

    Returns:
        jersey_counts, number_counts, border_counts,
        jersey_by_class, number_by_class, border_by_class,
        total_with, total_without,
        coverage_by_class
    """
    jersey_counts: Counter[str] = Counter()
    number_counts: Counter[str] = Counter()
    border_counts: Counter[bool] = Counter()
    jersey_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    number_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    border_by_class: dict[str, Counter[bool]] = defaultdict(
        Counter,
    )

    _cov_with: Counter[str] = Counter()
    _cov_without: Counter[str] = Counter()
    total_with = 0
    total_without = 0

    for record in records:
        cls = str(record.get("suffix", ""))
        meta = record.get("metadata")

        if meta and isinstance(meta, dict):
            total_with += 1
            _cov_with[cls] += 1

            jc = str(meta.get("jersey_color", "unknown"))
            nc = str(meta.get("number_color", "unknown"))
            border = meta.get("border", False)
            if not isinstance(border, bool):
                border = str(border).lower() in (
                    "true",
                    "yes",
                    "1",
                )

            jersey_counts[jc] += 1
            number_counts[nc] += 1
            border_counts[border] += 1
            jersey_by_class[cls][jc] += 1
            number_by_class[cls][nc] += 1
            border_by_class[cls][border] += 1
        else:
            total_without += 1
            _cov_without[cls] += 1

    all_classes = sorted(set(_cov_with.keys()) | set(_cov_without.keys()))
    coverage_by_class: dict[str, tuple[int, int]] = {}
    for cls in all_classes:
        coverage_by_class[cls] = (
            _cov_with[cls],
            _cov_without[cls],
        )

    return (
        jersey_counts,
        number_counts,
        border_counts,
        jersey_by_class,
        number_by_class,
        border_by_class,
        total_with,
        total_without,
        coverage_by_class,
    )


def plot_color_distribution(
    counts: Counter[str], title: str, output_path: Path
) -> None:
    """Horizontal bar chart of color distribution."""
    sorted_items = counts.most_common()
    if not sorted_items:
        logger.warning(f"No data for {title}, skipping")
        return

    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [COLOR_HEX.get(label, "#BDBDBD") for label in labels]

    height = max(4, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(
        labels,
        values,
        color=colors,
        edgecolor="#444444",
        linewidth=0.5,
    )
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, val in zip(bars, values, strict=True):
        x = bar.get_width() + max(values) * 0.01
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, str(val), va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_border_distribution(counts: Counter[bool], output_path: Path) -> None:
    """Pie chart of border vs no-border proportions."""
    labels = []
    values = []
    colors_list = []
    for has_border in [True, False]:
        if counts[has_border] > 0:
            lbl = "With Border" if has_border else "No Border"
            labels.append(lbl)
            values.append(counts[has_border])
            clr = "#1E88E5" if has_border else "#FB8C00"
            colors_list.append(clr)

    if not values:
        logger.warning("No border data, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        values,
        labels=labels,
        colors=colors_list,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title("Border Distribution")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_color_by_class_heatmap(
    by_class: dict[str, Counter[str]],
    title: str,
    output_path: Path,
) -> None:
    """Heatmap of color distribution per class (row-normalized)."""
    if not by_class:
        logger.warning(f"No data for {title}, skipping")
        return

    classes = sorted(by_class.keys())
    all_colors = sorted({c for cnts in by_class.values() for c in cnts})

    matrix = np.zeros((len(classes), len(all_colors)))
    for i, cls in enumerate(classes):
        row_total = sum(by_class[cls].values())
        for j, color in enumerate(all_colors):
            if row_total > 0:
                matrix[i, j] = by_class[cls][color] / row_total

    w = max(8, len(all_colors) * 0.8)
    h = max(6, len(classes) * 0.4)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(all_colors)))
    ax.set_xticklabels(
        all_colors,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Color")
    ax.set_ylabel("Class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Proportion")

    for i in range(len(classes)):
        for j in range(len(all_colors)):
            val = matrix[i, j]
            if val > 0:
                tc = "white" if val > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=tc,
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_border_by_class(by_class: dict[str, Counter[bool]], output_path: Path) -> None:
    """Stacked horizontal bar of border proportions per class."""
    if not by_class:
        logger.warning("No border-by-class data, skipping")
        return

    classes = sorted(by_class.keys())
    with_border = []
    without_border = []
    for cls in classes:
        total = sum(by_class[cls].values())
        if total > 0:
            with_border.append(by_class[cls][True] / total)
            without_border.append(by_class[cls][False] / total)
        else:
            with_border.append(0)
            without_border.append(0)

    h = max(4, len(classes) * 0.4)
    fig, ax = plt.subplots(figsize=(10, h))
    y_pos = range(len(classes))
    ax.barh(
        y_pos,
        with_border,
        label="With Border",
        color="#1E88E5",
    )
    ax.barh(
        y_pos,
        without_border,
        left=with_border,
        label="No Border",
        color="#FB8C00",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Proportion")
    ax.set_title("Border Distribution by Class")
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_color_combinations(
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Heatmap of jersey_color x number_color combos."""
    combo_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        meta = record.get("metadata")
        if meta and isinstance(meta, dict):
            jc = str(meta.get("jersey_color", "unknown"))
            nc = str(meta.get("number_color", "unknown"))
            combo_counts[(jc, nc)] += 1

    if not combo_counts:
        logger.warning("No color combo data, skipping")
        return

    jersey_colors = sorted({k[0] for k in combo_counts})
    number_colors = sorted({k[1] for k in combo_counts})

    matrix = np.zeros((len(jersey_colors), len(number_colors)))
    for i, jc in enumerate(jersey_colors):
        for j, nc in enumerate(number_colors):
            matrix[i, j] = combo_counts.get((jc, nc), 0)

    w = max(8, len(number_colors) * 0.8)
    h = max(6, len(jersey_colors) * 0.6)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(number_colors)))
    ax.set_xticklabels(
        number_colors,
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(range(len(jersey_colors)))
    ax.set_yticklabels(jersey_colors, fontsize=9)
    ax.set_xlabel("Number Color")
    ax.set_ylabel("Jersey Color")
    ax.set_title("Jersey x Number Color Combinations")
    fig.colorbar(im, ax=ax, label="Count")

    max_val = matrix.max()
    for i in range(len(jersey_colors)):
        for j in range(len(number_colors)):
            val = int(matrix[i, j])
            if val > 0:
                tc = "white" if val > max_val * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=tc,
                )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_annotation_coverage(
    coverage_by_class: dict[str, tuple[int, int]],
    output_path: Path,
) -> None:
    """Bar chart showing per-class annotation coverage."""
    if not coverage_by_class:
        logger.warning("No coverage data, skipping plot")
        return

    classes = sorted(coverage_by_class.keys())
    pct_covered = []
    for cls in classes:
        w, wo = coverage_by_class[cls]
        total = w + wo
        pct = w / total * 100 if total > 0 else 0
        pct_covered.append(pct)

    def _bar_color(p: float) -> str:
        if p >= 90:
            return "#43A047"
        if p >= 50:
            return "#FDD835"
        return "#E53935"

    w = max(10, len(classes) * 0.3)
    fig, ax = plt.subplots(figsize=(w, 5))
    colors = [_bar_color(p) for p in pct_covered]
    ax.bar(
        range(len(classes)),
        pct_covered,
        color=colors,
        edgecolor="#444444",
        linewidth=0.5,
    )
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Annotation Coverage by Class")
    ax.set_ylim(0, 105)
    ax.axhline(
        y=100,
        color="#888888",
        linestyle="--",
        linewidth=0.5,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metadata statistics and plots"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to analyze (default: train)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metadata_stats"),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    if not split_dir.exists():
        logger.error(f"Split not found: {split_dir}")
        sys.exit(1)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading records from {split_dir}...")
    records = load_records(split_dir)
    logger.info(f"Loaded {len(records)} records")

    (
        jersey_counts,
        number_counts,
        border_counts,
        jersey_by_class,
        number_by_class,
        border_by_class,
        total_with,
        total_without,
        coverage_by_class,
    ) = collect_stats(records)

    total = total_with + total_without
    if total > 0:
        pct = total_with / total * 100
        logger.info(f"Metadata coverage: {total_with}/{total} ({pct:.1f}%)")
    else:
        logger.info("No records found")

    # Generate all plots
    jc_path = output_dir / "jersey_color_distribution.png"
    plot_color_distribution(
        jersey_counts,
        "Jersey Color Distribution",
        jc_path,
    )
    nc_path = output_dir / "number_color_distribution.png"
    plot_color_distribution(
        number_counts,
        "Number Color Distribution",
        nc_path,
    )
    plot_border_distribution(
        border_counts,
        output_dir / "border_distribution.png",
    )
    plot_color_by_class_heatmap(
        jersey_by_class,
        "Jersey Color by Class",
        output_dir / "jersey_color_by_class.png",
    )
    plot_color_by_class_heatmap(
        number_by_class,
        "Number Color by Class",
        output_dir / "number_color_by_class.png",
    )
    plot_border_by_class(
        border_by_class,
        output_dir / "border_by_class.png",
    )
    plot_color_combinations(
        records,
        output_dir / "color_combinations.png",
    )
    plot_annotation_coverage(
        coverage_by_class,
        output_dir / "annotation_coverage.png",
    )

    logger.info(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
