"""Collator for VLM fine-tuning batches."""

from __future__ import annotations

from typing import Any

from PIL import Image


class VLMCollator:
    """Collates VLM dataset items into processor-ready batches.

    Applies chat template, runs processor on images+text, creates labels
    with prompt tokens masked to ``-100`` (only answer tokens contribute to loss).

    Args:
        processor: HuggingFace processor (tokenizer + image processor).
        max_length: Maximum sequence length for tokenization.
    """

    IGNORE_INDEX = -100

    def __init__(self, processor: Any, max_length: int = 384) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[tuple[Image.Image, str, str]]) -> dict[str, Any]:
        """Collate a list of (image, prompt, answer) tuples into a model-ready batch.

        Returns dict with ``input_ids``, ``attention_mask``, ``pixel_values``,
        and ``labels`` (prompt tokens masked to -100).
        """
        images: list[Image.Image] = []
        full_texts: list[str] = []
        prompt_texts: list[str] = []

        for image, prompt, answer in batch:
            images.append(image)

            # Build chat messages with answer
            messages_with_answer: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            full_text: str = self.processor.apply_chat_template(
                messages_with_answer, tokenize=False
            )
            full_texts.append(full_text)

            # Build prompt-only messages (for masking)
            messages_prompt_only: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            prompt_text: str = self.processor.apply_chat_template(
                messages_prompt_only,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_texts.append(prompt_text)

        # Process full sequences (prompt + answer)
        # NOTE: truncation is disabled because SmolVLM2 expands <image> into
        # many tokens (~1377); truncating would cause a mismatch between the
        # image token count in text vs input_ids.  Sequences are short (image
        # tokens + short prompt + 1-2 digit answer) so no OOM risk.
        batch_images = [[img] for img in images]
        inputs = self.processor(
            text=full_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize prompt-only to find where answer starts
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )

        # Build labels: clone input_ids, mask prompt portion to -100
        labels = inputs["input_ids"].clone()
        for i in range(len(batch)):
            # Count non-padding tokens in prompt-only
            prompt_mask = prompt_inputs["attention_mask"][i]
            prompt_len = prompt_mask.sum().item()
            # Mask all prompt tokens (only answer tokens contribute to loss)
            labels[i, :prompt_len] = self.IGNORE_INDEX
            # Also mask padding tokens
            padding_mask = inputs["attention_mask"][i] == 0
            labels[i, padding_mask] = self.IGNORE_INDEX

        inputs["labels"] = labels
        return dict(inputs)
