"""SmolVLM2-based classification inferencer."""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger
from PIL import Image

from classifier_training.inference.base import BaseClassificationInferencer
from classifier_training.schemas.annotation import ClassificationPrediction

VLM_PROMPT = (
    "What number is on this basketball jersey? "
    "Reply with just the number, nothing else."
)


def _parse_vlm_response(
    text: str, class_to_idx: dict[str, int], idx_to_class: dict[int, str]
) -> list[ClassificationPrediction]:
    """Parse VLM text response into classification predictions.

    The VLM returns a text answer.  We try to match it to a known class
    label.  If matched, confidence is 1.0 (single-label, no distribution).
    If not matched, we return the empty-string class at confidence 1.0
    as a fallback.
    """
    cleaned = text.strip().strip('"').strip("'").strip(".")
    # Try direct match
    if cleaned in class_to_idx:
        idx = class_to_idx[cleaned]
        return [ClassificationPrediction(class_id=idx, label=cleaned, confidence=1.0)]

    # Try extracting a number from the response
    match = re.search(r"\b(\d{1,2})\b", cleaned)
    if match:
        number = match.group(1)
        if number in class_to_idx:
            idx = class_to_idx[number]
            return [
                ClassificationPrediction(class_id=idx, label=number, confidence=1.0)
            ]

    # Fallback: empty-string class (index 0) if available
    logger.debug(f"VLM response not matched to any class: {text!r}")
    if "" in class_to_idx:
        idx = class_to_idx[""]
        return [ClassificationPrediction(class_id=idx, label="", confidence=1.0)]

    return []


class SmolVLM2ClassificationInferencer(BaseClassificationInferencer):
    """Classification inferencer using SmolVLM2 vision-language model.

    Asks the VLM to identify the jersey number, then maps the text
    response to a class label from ``labels_mapping.json``.

    Args:
        model_name: HuggingFace model identifier.
        labels_mapping_path: Path to ``labels_mapping.json``.
        batch_size: Max images per batch for generation.
        device: Device string (``"cuda"``, ``"mps"``, ``"cpu"``).
            Auto-detected if *None*.
    """

    def __init__(
        self,
        model_name: str,
        labels_mapping_path: str | Path,
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        labels_mapping_path = Path(labels_mapping_path)
        with open(labels_mapping_path) as f:
            mapping = json.load(f)

        self.class_to_idx: dict[str, int] = mapping["class_to_idx"]
        self.idx_to_class: dict[int, str] = {
            int(k): v for k, v in mapping["idx_to_class"].items()
        }
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        dtype_map = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float16}
        dtype = dtype_map.get(device, torch.float16)

        # Try flash_attention_2 on CUDA, fall back to eager if not installed
        attn_impl = "eager"
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                logger.info("flash_attn not installed, using eager attention")

        logger.info(f"Loading VLM {model_name} on {device} (attn={attn_impl})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            _attn_implementation=attn_impl,
        ).to(device)

        if device == "cuda":
            self.model = torch.compile(self.model)

    def predict(self, image: Image.Image) -> list[ClassificationPrediction]:
        """Single image inference via VLM."""
        results = self.predict_batch([image])
        return results[0]

    def predict_batch(
        self, images: list[Image.Image]
    ) -> list[list[ClassificationPrediction]]:
        """Batched VLM inference."""
        if not images:
            return []

        results: list[list[ClassificationPrediction]] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]
            batch_results = self._generate_batch(batch_images)
            results.extend(batch_results)
        return results

    def _generate_batch(
        self, images: list[Image.Image]
    ) -> list[list[ClassificationPrediction]]:
        """Generate predictions for a batch of images."""
        import torch

        # Build per-image messages
        all_messages = []
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.convert("RGB")},
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                }
            ]
            all_messages.append(messages)

        # Process each conversation through chat template
        texts = [
            self.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            for msgs in all_messages
        ]

        # Batch-process with padding
        # Each text references one image, so images must be list-of-lists
        batch_images = [[img.convert("RGB")] for img in images]
        inputs = self.processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=64)

        # Decode each output individually
        results = []
        prompt_len = inputs["input_ids"].shape[-1]
        for i in range(len(images)):
            generated = output_ids[i][prompt_len:]
            text = self.processor.decode(generated, skip_special_tokens=True)
            preds = _parse_vlm_response(text, self.class_to_idx, self.idx_to_class)
            results.append(preds)
        return results
