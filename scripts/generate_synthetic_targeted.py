#!/usr/bin/env python3
"""Generate targeted synthetic jersey number images for specific color combos.

Unlike generate_synthetic.py (which fills general class gaps), this script
generates images for specific jersey_color x number_color combinations that
are known to be absent or rare in training data.

Usage::

    python scripts/generate_synthetic_targeted.py \
        --config configs/synthetic_targeted.yaml --dry-run
    python scripts/generate_synthetic_targeted.py \
        --config configs/synthetic_targeted.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

# Add project root to path so we can import classifier_training
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier_training.synthetic.renderer import (  # noqa: E402
    _COLOR_BY_NAME,
    JerseyNumberRenderer,
)
from classifier_training.synthetic.writer import SyntheticWriter  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate targeted synthetic images for specific color combos"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file with targets list",
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

    output_dir = Path(str(config["output_dir"]))
    image_size = int(config["image_size"])  # type: ignore[arg-type]
    font_dir = PROJECT_ROOT / str(config["font_dir"])
    seed = int(config["seed"])  # type: ignore[arg-type]
    targets: list[dict[str, object]] = config["targets"]  # type: ignore[assignment]

    # Validate and print plan
    total = 0
    for target in targets:
        label = str(target["label"])
        combos: list[dict[str, object]] = target["combos"]  # type: ignore[assignment]
        for combo in combos:
            jc = str(combo["jersey_color"])
            nc = str(combo["number_color"])
            count = int(combo["count"])  # type: ignore[arg-type]

            if jc not in _COLOR_BY_NAME:
                logger.error(
                    f"Unknown jersey_color {jc!r}. "
                    f"Valid: {sorted(_COLOR_BY_NAME)}"
                )
                sys.exit(1)
            if nc not in _COLOR_BY_NAME:
                logger.error(
                    f"Unknown number_color {nc!r}. "
                    f"Valid: {sorted(_COLOR_BY_NAME)}"
                )
                sys.exit(1)

            logger.info(
                f"  label={label!r}  jersey={jc}  number={nc}  count={count}"
            )
            total += count

    logger.info(
        f"Targeted generation plan: {len(targets)} labels, {total} total images"
    )

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

    global_index = 0
    for target in targets:
        label = str(target["label"])
        combos = target["combos"]  # type: ignore[assignment]
        for combo in combos:
            jc = str(combo["jersey_color"])
            nc = str(combo["number_color"])
            count = int(combo["count"])  # type: ignore[arg-type]

            bg_rgb = _COLOR_BY_NAME[jc]
            text_rgb = _COLOR_BY_NAME[nc]

            for _ in range(count):
                img, metadata = renderer.render(
                    label,
                    bg_color=bg_rgb,
                    text_color=text_rgb,
                )
                writer.write_image(img, label, global_index, metadata=metadata)
                global_index += 1

    ann_path = writer.flush()
    logger.info(f"Done! Generated {writer.num_written} images in {output_dir}")
    logger.info(f"Annotations: {ann_path}")


if __name__ == "__main__":
    main()
