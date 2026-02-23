#!/usr/bin/env python3
"""Evaluate classification models on a dataset split.

Runs ONNX and/or VLM inferencers on val/test images, saves per-image
ClassificationAnnotation JSONs, and prints top-1/top-5 accuracy.

Usage::

    # Both methods (using --model-dir convenience arg)
    python scripts/evaluate_classifiers.py \\
        --data-root /path/to/dataset \\
        --model-dir models/best \\
        --method both

    # ONNX only (explicit paths)
    python scripts/evaluate_classifiers.py \\
        --data-root /path/to/dataset \\
        --model-path models/best/model.onnx \\
        --labels-mapping models/best/labels_mapping.json \\
        --method onnx

    # VLM only
    python scripts/evaluate_classifiers.py \\
        --data-root /path/to/dataset \\
        --labels-mapping models/best/labels_mapping.json \\
        --method vlm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from classifier_training.data.utils import get_files  # noqa: E402
from classifier_training.inference.base import (  # noqa: E402
    BaseClassificationInferencer,
)
from classifier_training.io.annotation import (  # noqa: E402
    ClassificationAnnotationWriter,
)
from classifier_training.schemas.annotation import (  # noqa: E402
    ClassificationAnnotation,
)
from classifier_training.schemas.info import AnnotationInfo  # noqa: E402


def load_samples(data_root: Path, split: str) -> list[tuple[Path, str]]:
    """Load (image_path, ground_truth_label) pairs from JSONL files."""
    split_dir = data_root / split
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        sys.exit(1)

    ann_files = get_files(split_dir, (".jsonl",))
    if not ann_files:
        logger.error(f"No .jsonl files found under {split_dir}")
        sys.exit(1)

    samples: list[tuple[Path, str]] = []
    for ann_path in ann_files:
        ann_dir = ann_path.parent
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                img_path = ann_dir / record["image"]
                label = record["suffix"]
                samples.append((img_path, label))

    logger.info(f"Loaded {len(samples)} samples from {split} split")
    return samples


def evaluate_inferencer(
    inferencer: BaseClassificationInferencer,
    samples: list[tuple[Path, str]],
    idx_to_class: dict[int, str],
    output_dir: Path,
    source_name: str,
    batch_size: int,
    limit: int | None = None,
) -> dict[str, object]:
    """Run inferencer on samples, write annotations, compute metrics."""
    writer = ClassificationAnnotationWriter(output_dir)
    categories = {int(k): v for k, v in idx_to_class.items()}

    if limit is not None:
        samples = samples[:limit]

    top1_correct = 0
    top5_correct = 0
    total = 0
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)

    # Process in batches
    for start in tqdm(
        range(0, len(samples), batch_size),
        desc=source_name,
        total=(len(samples) + batch_size - 1) // batch_size,
    ):
        batch_samples = samples[start : start + batch_size]
        batch_images = []
        batch_labels = []
        batch_paths = []

        for img_path, label in batch_samples:
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            batch_images.append(Image.open(img_path).convert("RGB"))
            batch_labels.append(label)
            batch_paths.append(img_path)

        if not batch_images:
            continue

        batch_predictions = inferencer.predict_batch(batch_images)

        for img_path, gt_label, preds in zip(
            batch_paths, batch_labels, batch_predictions, strict=True
        ):
            img = Image.open(img_path)
            annotation = ClassificationAnnotation(
                filename=img_path.name,
                categories=categories,
                info=AnnotationInfo(
                    annotations_source=source_name,
                    image_width=img.width,
                    image_height=img.height,
                ),
                ground_truth=gt_label,
                predictions=preds,
            )
            writer.write(annotation)

            # Compute metrics
            total += 1
            per_class_total[gt_label] += 1

            pred_labels = [p.label for p in preds]
            if pred_labels and pred_labels[0] == gt_label:
                top1_correct += 1
                per_class_correct[gt_label] += 1

            if gt_label in pred_labels[:5]:
                top5_correct += 1

    top1_acc = top1_correct / total if total > 0 else 0.0
    top5_acc = top5_correct / total if total > 0 else 0.0

    per_class_accuracy = {}
    for cls in sorted(per_class_total.keys()):
        cls_total = per_class_total[cls]
        cls_correct = per_class_correct.get(cls, 0)
        per_class_accuracy[cls] = {
            "correct": cls_correct,
            "total": cls_total,
            "accuracy": cls_correct / cls_total if cls_total > 0 else 0.0,
        }

    return {
        "source": source_name,
        "total": total,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_acc,
        "top5_correct": top5_correct,
        "top5_accuracy": top5_acc,
        "per_class": per_class_accuracy,
    }


def print_results(results: list[dict[str, object]]) -> None:
    """Print a Rich table summarizing evaluation results."""
    console = Console()

    # Summary table
    table = Table(title="Evaluation Results")
    table.add_column("Method", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Top-1 Acc", justify="right", style="green")
    table.add_column("Top-5 Acc", justify="right", style="green")

    for r in results:
        table.add_row(
            str(r["source"]),
            str(r["total"]),
            f"{r['top1_accuracy']:.1%}",
            f"{r['top5_accuracy']:.1%}",
        )

    console.print(table)

    # Per-class breakdown for each method
    for r in results:
        per_class = r.get("per_class", {})
        if not per_class:
            continue
        if not isinstance(per_class, dict):
            continue

        cls_table = Table(title=f"Per-class Accuracy: {r['source']}")
        cls_table.add_column("Class", style="cyan")
        cls_table.add_column("Correct", justify="right")
        cls_table.add_column("Total", justify="right")
        cls_table.add_column("Accuracy", justify="right", style="green")

        for cls in sorted(per_class.keys()):
            info = per_class[cls]
            if not isinstance(info, dict):
                continue
            cls_label = cls if cls else '""'
            cls_table.add_row(
                cls_label,
                str(info["correct"]),
                str(info["total"]),
                f"{info['accuracy']:.1%}",
            )

        console.print(cls_table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classifiers on a dataset split"
    )
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Dataset root directory"
    )
    parser.add_argument(
        "--split", type=str, default="valid", help="Split to evaluate (default: valid)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing model.onnx and labels_mapping.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to ONNX model file (overrides --model-dir)",
    )
    parser.add_argument(
        "--labels-mapping",
        type=Path,
        default=None,
        help="Path to labels_mapping.json (overrides --model-dir)",
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        help="HuggingFace VLM model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=8,
        help="Batch size for VLM inference (default: 8)",
    )
    parser.add_argument(
        "--method",
        choices=["onnx", "vlm", "both"],
        default="both",
        help="Which inferencer(s) to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N samples",
    )
    args = parser.parse_args()

    # Resolve model-dir convenience arg
    if args.model_dir is not None:
        if args.model_path is None:
            args.model_path = args.model_dir / "model.onnx"
        if args.labels_mapping is None:
            args.labels_mapping = args.model_dir / "labels_mapping.json"

    if args.labels_mapping is None:
        logger.error("--labels-mapping or --model-dir is required")
        sys.exit(1)

    with open(args.labels_mapping) as f:
        mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}

    samples = load_samples(args.data_root, args.split)
    results: list[dict[str, object]] = []

    # ONNX evaluation
    if args.method in ("onnx", "both"):
        if args.model_path is None:
            logger.error("--model-path is required for ONNX evaluation")
            sys.exit(1)

        from classifier_training.inference.onnx_inferencer import (
            ONNXClassificationInferencer,
        )

        logger.info("Running ONNX evaluation...")
        onnx_inferencer = ONNXClassificationInferencer(
            model_path=args.model_path,
            labels_mapping_path=args.labels_mapping,
            top_k=5,
        )
        onnx_result = evaluate_inferencer(
            onnx_inferencer,
            samples,
            idx_to_class,
            args.output_dir / "onnx",
            "onnx-resnet18",
            args.batch_size,
            args.limit,
        )
        results.append(onnx_result)

    # VLM evaluation
    if args.method in ("vlm", "both"):
        from classifier_training.inference.vlm_inferencer import (
            SmolVLM2ClassificationInferencer,
        )

        logger.info("Running VLM evaluation...")
        vlm_inferencer = SmolVLM2ClassificationInferencer(
            model_name=args.vlm_model_name,
            labels_mapping_path=args.labels_mapping,
            batch_size=args.vlm_batch_size,
        )
        vlm_result = evaluate_inferencer(
            vlm_inferencer,
            samples,
            idx_to_class,
            args.output_dir / "vlm",
            "smolvlm2-2.2b",
            args.vlm_batch_size,
            args.limit,
        )
        results.append(vlm_result)

    # Save summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
