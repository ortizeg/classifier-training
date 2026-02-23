#!/usr/bin/env python3
"""Annotate JSONL training data with jersey visual metadata using SmolVLM2.

Adds jersey_color, number_color, and border fields to each annotation record
using a vision-language model. Resumable — skips records that already have metadata.

Usage::

    python scripts/annotate_metadata.py --data-root /path/to/dataset
    python scripts/annotate_metadata.py --data-root /path/to/dataset --mode multi
    python scripts/annotate_metadata.py --data-root /path/to/dataset --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from loguru import logger

# Add project root to path so we can import classifier_training
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier_training.data.utils import get_files  # noqa: E402

VALID_COLORS = frozenset(
    {
        "white",
        "black",
        "red",
        "blue",
        "navy",
        "yellow",
        "green",
        "purple",
        "orange",
        "grey",
        "maroon",
        "teal",
        "pink",
    }
)

SINGLE_PROMPT_SYSTEM = (
    "You analyze basketball jersey images. Respond with valid JSON only."
)

SINGLE_PROMPT_USER = (
    "Identify: 1) jersey background color, 2) number color, "
    "3) does number have a border/outline?\n"
    'Reply ONLY with: {{"jersey_color": "<color>", '
    '"number_color": "<color>", "border": true/false}}\n'
    "Valid colors: white, black, red, blue, navy, yellow, green, "
    "purple, orange, grey, maroon, teal, pink"
)

MULTI_PROMPT_JERSEY = (
    "What is the main color of this basketball jersey? "
    "Reply with ONE word from: white, black, red, blue, navy, yellow, "
    "green, purple, orange, grey, maroon, teal, pink"
)
MULTI_PROMPT_NUMBER = (
    "What color is the number on this jersey? "
    "Reply with ONE word from: white, black, red, blue, navy, yellow, "
    "green, purple, orange, grey, maroon, teal, pink"
)
MULTI_PROMPT_BORDER = (
    "Does the number have a visible border or outline around it? "
    "Reply yes or no."
)


def parse_single_response(text: str) -> dict[str, object] | None:
    """Parse JSON response from single-mode prompt.

    Tries direct json.loads first, then regex extraction.
    Returns None if parsing fails.
    """
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        return _validate_single_data(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try regex extraction
    match = re.search(r"\{[^}]+\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_single_data(data)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _validate_single_data(data: object) -> dict[str, object]:
    """Validate and normalize parsed single-mode JSON data."""
    if not isinstance(data, dict):
        raise ValueError("Expected dict")

    jersey_color = _normalize_color(str(data.get("jersey_color", "")))
    number_color = _normalize_color(str(data.get("number_color", "")))
    border = _parse_border(data.get("border", False))

    return {
        "jersey_color": jersey_color,
        "number_color": number_color,
        "border": border,
    }


def _normalize_color(raw: str) -> str:
    """Normalize a color string, returning 'unknown' if not in valid set."""
    color = raw.lower().strip()
    return color if color in VALID_COLORS else "unknown"


def parse_color_response(text: str) -> str:
    """Parse a single-word color response from multi-mode prompt."""
    return _normalize_color(text)


def _parse_border(value: object) -> bool:
    """Parse a border value from various formats."""
    if isinstance(value, bool):
        return value
    s = str(value).lower().strip()
    return s in ("true", "yes", "y", "1")


def parse_border_response(text: str) -> bool:
    """Parse yes/no border response from multi-mode prompt."""
    return _parse_border(text)


def _auto_device() -> tuple[str, str]:
    """Auto-detect best device and dtype.

    Returns (device, dtype_str) tuple.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "float16"
    return "cpu", "float16"


def _load_model(
    model_name: str, device: str, dtype_str: str
) -> tuple[object, object]:
    """Load SmolVLM2 model and processor.

    Returns (processor, model) tuple.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(dtype_str, torch.float16)

    logger.info(f"Loading model {model_name} on {device} ({dtype_str})")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        _attn_implementation="eager",
    ).to(device)

    return processor, model


def _generate(
    processor: object,
    model: object,
    image: object,
    user_text: str,
    system_text: str | None = None,
    device: str = "cpu",
) -> str:
    """Generate text from image + prompt using the loaded model."""
    import torch

    messages: list[dict[str, object]] = []
    if system_text:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        })

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        }
    )

    inputs = processor.apply_chat_template(  # type: ignore[union-attr]
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128)  # type: ignore[union-attr]

    # Decode only newly generated tokens
    prompt_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0][prompt_len:]
    return processor.decode(generated, skip_special_tokens=True)  # type: ignore[union-attr]


def annotate_single(
    processor: object,
    model: object,
    image: object,
    device: str,
) -> dict[str, object] | None:
    """Annotate a single image using single-mode (one JSON prompt)."""
    response = _generate(
        processor, model, image, SINGLE_PROMPT_USER, SINGLE_PROMPT_SYSTEM, device
    )
    return parse_single_response(response)


def annotate_multi(
    processor: object,
    model: object,
    image: object,
    device: str,
) -> dict[str, object]:
    """Annotate a single image using multi-mode (3 separate prompts)."""
    jersey_resp = _generate(processor, model, image, MULTI_PROMPT_JERSEY, device=device)
    number_resp = _generate(processor, model, image, MULTI_PROMPT_NUMBER, device=device)
    border_resp = _generate(processor, model, image, MULTI_PROMPT_BORDER, device=device)

    return {
        "jersey_color": parse_color_response(jersey_resp),
        "number_color": parse_color_response(number_resp),
        "border": parse_border_response(border_resp),
    }


SAVE_INTERVAL = 100  # Save progress every N new annotations


def _atomic_write(ann_path: Path, records: list[dict[str, object]]) -> None:
    """Atomically write records back to JSONL file."""
    updated_lines = [
        json.dumps(r, ensure_ascii=False) for r in records
    ]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=ann_path.parent,
            suffix=".jsonl.tmp",
            delete=False,
        ) as tmp_fd:
            tmp_path = Path(tmp_fd.name)
            tmp_fd.write("\n".join(updated_lines) + "\n")
        tmp_path.replace(ann_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def process_annotation_file(
    ann_path: Path,
    processor: object,
    model: object,
    device: str,
    mode: str,
    dry_run: bool,
    limit: int | None,
) -> tuple[int, int]:
    """Process a single annotation file, adding metadata in-place.

    Saves progress every SAVE_INTERVAL new annotations so work
    is not lost if the process is interrupted.

    Returns (annotated_count, skipped_count).
    """
    from PIL import Image

    ann_dir = ann_path.parent

    # Read all records
    with open(ann_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    records = [json.loads(line) for line in lines]
    annotated = 0
    skipped = 0
    unsaved = 0  # New annotations since last save

    for _i, record in enumerate(records):
        if limit is not None and annotated >= limit:
            break

        # Skip already annotated records
        if "metadata" in record:
            skipped += 1
            continue

        img_path = ann_dir / record["image"]
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            skipped += 1
            continue

        if dry_run:
            img_name = record["image"]
            suffix = record["suffix"]
            logger.info(
                f"[DRY RUN] Would annotate: {img_name} "
                f"(suffix={suffix})"
            )
            annotated += 1
            continue

        image = Image.open(img_path).convert("RGB")

        if mode == "single":
            metadata = annotate_single(
                processor, model, image, device,
            )
            if metadata is None:
                logger.warning(
                    f"Failed to parse response for "
                    f"{record['image']}, skipping"
                )
                skipped += 1
                continue
        else:
            metadata = annotate_multi(
                processor, model, image, device,
            )

        record["metadata"] = metadata
        annotated += 1
        unsaved += 1

        if annotated % 10 == 0:
            logger.info(
                f"  [{annotated}] {ann_path.name}"
            )

        # Periodic save to preserve progress
        if unsaved >= SAVE_INTERVAL and not dry_run:
            _atomic_write(ann_path, records)
            logger.info(
                f"  Checkpoint: saved {annotated} "
                f"annotations to {ann_path.name}"
            )
            unsaved = 0

    # Final save for any remaining unsaved work
    if unsaved > 0 and not dry_run:
        _atomic_write(ann_path, records)
        logger.info(
            f"Updated {ann_path} "
            f"({annotated} new annotations)"
        )

    return annotated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate JSONL data with jersey visual metadata using SmolVLM2"
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
        help="Dataset split to annotate (default: train)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Annotation mode: single (1 JSON prompt) or multi (3 prompts)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be annotated without writing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N unannotated records per file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        help="HuggingFace model name",
    )
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        sys.exit(1)

    ann_files = get_files(split_dir, (".jsonl",))
    if not ann_files:
        logger.error(f"No .jsonl files found under {split_dir}")
        sys.exit(1)

    logger.info(
        f"Found {len(ann_files)} annotation file(s) under {split_dir} "
        f"(mode={args.mode}, dry_run={args.dry_run})"
    )

    # Load model (skip for dry run)
    processor, model = None, None
    device = "cpu"
    if not args.dry_run:
        device, dtype_str = _auto_device()
        processor, model = _load_model(args.model_name, device, dtype_str)

    total_annotated = 0
    total_skipped = 0

    for ann_path in ann_files:
        logger.info(f"Processing {ann_path}...")
        annotated, skipped = process_annotation_file(
            ann_path, processor, model, device, args.mode, args.dry_run, args.limit
        )
        total_annotated += annotated
        total_skipped += skipped

    logger.info(
        f"Done! Annotated: {total_annotated}, "
        f"Skipped (already done or errors): {total_skipped}"
    )


if __name__ == "__main__":
    main()
