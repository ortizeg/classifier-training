#!/usr/bin/env python3
"""Generate synthetic jersey number images for underrepresented classes.

Reads real training annotations to identify classes below the threshold,
then renders synthetic images to bring them up to target_per_class.

Usage::

    python scripts/generate_synthetic.py --config configs/synthetic_default.yaml
    python scripts/generate_synthetic.py --config configs/synthetic_default.yaml \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import string
import sys
from collections import Counter
from pathlib import Path

import yaml
from loguru import logger

# Add project root to path so we can import classifier_training
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier_training.data.utils import get_files  # noqa: E402
from classifier_training.synthetic.renderer import JerseyNumberRenderer  # noqa: E402
from classifier_training.synthetic.writer import SyntheticWriter  # noqa: E402


def _count_real_samples(train_root: Path) -> Counter[str]:
    """Count real samples per class from all annotation files under train_root."""
    counts: Counter[str] = Counter()
    ann_files = get_files(train_root, (".jsonl",))
    for ann_path in ann_files:
        # Skip synthetic annotations to count only real data
        if "synthetic" in str(ann_path):
            continue
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                counts[record["suffix"]] += 1
    return counts


def _resolve_config_vars(config: dict[str, object]) -> dict[str, object]:
    """Resolve ${data_root} variable references in config values."""
    data_root = str(config.get("data_root", ""))
    resolved = {}
    for key, val in config.items():
        if isinstance(val, str) and "${data_root}" in val:
            resolved[key] = val.replace("${data_root}", data_root)
        else:
            resolved[key] = val
    return resolved


def _is_renderable(label: str) -> bool:
    """Check if a label contains only digits (renderable with our fonts).

    Skips empty string and any non-digit labels.
    """
    return len(label) > 0 and all(c in string.digits for c in label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic jersey number images"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generation plan without writing images",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = _resolve_config_vars(yaml.safe_load(f))

    data_root = Path(str(config["data_root"]))
    output_dir = Path(str(config["output_dir"]))
    image_size = int(config["image_size"])  # type: ignore[arg-type]
    target_per_class = int(config["target_per_class"])  # type: ignore[arg-type]
    threshold = int(config["threshold"])  # type: ignore[arg-type]
    font_dir = PROJECT_ROOT / str(config["font_dir"])
    seed = int(config["seed"])  # type: ignore[arg-type]

    train_root = data_root / "train"
    if not train_root.exists():
        logger.error(f"Train root not found: {train_root}")
        sys.exit(1)

    # Count real samples per class
    real_counts = _count_real_samples(train_root)
    all_classes = sorted(real_counts.keys())

    total_real = sum(real_counts.values())
    logger.info(f"Found {len(all_classes)} classes, {total_real} total real samples")

    # Identify underrepresented classes
    generation_plan: list[
        tuple[str, int, int]
    ] = []  # (label, real_count, num_to_generate)
    for cls in all_classes:
        real = real_counts[cls]
        if real < threshold and _is_renderable(cls):
            num_synth = max(0, target_per_class - real)
            if num_synth > 0:
                generation_plan.append((cls, real, num_synth))

    if not generation_plan:
        logger.info("No classes below threshold — nothing to generate.")
        return

    # Print generation plan
    total_synth = sum(n for _, _, n in generation_plan)
    logger.info(
        f"Will generate {total_synth} synthetic images "
        f"for {len(generation_plan)} classes (threshold={threshold}, "
        f"target={target_per_class})"
    )
    print(f"\n{'Class':>6}  {'Real':>5}  {'Synth':>6}  {'Total':>6}")
    print("-" * 30)
    for cls, real, num_synth in generation_plan:
        print(f"{cls:>6}  {real:>5}  {num_synth:>6}  {real + num_synth:>6}")
    print("-" * 30)
    print(f"{'TOTAL':>6}  {'':>5}  {total_synth:>6}")
    print()

    if args.dry_run:
        logger.info("Dry run — no images generated.")
        return

    # Generate
    renderer = JerseyNumberRenderer(
        font_dir=font_dir,
        image_size=image_size,
        seed=seed,
    )
    writer = SyntheticWriter(output_dir=output_dir)

    for cls, _real, num_synth in generation_plan:
        for i in range(num_synth):
            img = renderer.render(cls)
            writer.write_image(img, cls, i)

    ann_path = writer.flush()
    logger.info(f"Done! Generated {writer.num_written} images in {output_dir}")
    logger.info(f"Annotations: {ann_path}")

    # Print summary of all classes (real + synthetic)
    print(f"\n{'Class':>6}  {'Real':>5}  {'Synth':>6}  {'Total':>6}")
    print("=" * 30)
    for cls in all_classes:
        real = real_counts[cls]
        synth = 0
        for plan_cls, _, num_synth in generation_plan:
            if plan_cls == cls:
                synth = num_synth
                break
        total = real + synth
        marker = " *" if synth > 0 else ""
        print(f"{cls:>6}  {real:>5}  {synth:>6}  {total:>6}{marker}")
    print("=" * 30)
    grand_real = sum(real_counts.values())
    grand_total = grand_real + total_synth
    print(f"{'TOTAL':>6}  {grand_real:>5}  {total_synth:>6}  {grand_total:>6}")
    print("\n* = synthetic data generated")


if __name__ == "__main__":
    main()
