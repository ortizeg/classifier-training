"""Unit tests for VLMCollator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from classifier_training.data.vlm_collator import VLMCollator


@pytest.fixture()
def mock_processor() -> MagicMock:
    """Create a mock HF processor for testing collator logic."""
    processor = MagicMock()

    def fake_apply_chat_template(
        messages: list[dict[str, object]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        """Simulate chat template application."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                text_parts = [
                    c["text"]
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                ]
                parts.append(f"<{role}>{''.join(text_parts)}</{role}>")
            elif isinstance(content, str):
                parts.append(f"<{role}>{content}</{role}>")
        result = "".join(parts)
        if add_generation_prompt:
            result += "<assistant>"
        return result

    processor.apply_chat_template = fake_apply_chat_template

    call_count = {"n": 0}

    def fake_call(
        text: list[str] | None = None,
        images: list[list[Image.Image]] | None = None,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 384,
    ) -> dict[str, torch.Tensor]:
        """Simulate processor tokenization.

        First call (full text with answer) returns seq_len=20.
        Second call (prompt-only) returns seq_len=15 with 5 padding tokens,
        so prompt_len=15 and answer tokens (positions 15-19) are unmasked.
        """
        batch_size = len(text) if text else 1
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:
            # Full text (prompt + answer)
            seq_len = 20
            input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        else:
            # Prompt-only: shorter content, padded to same length
            seq_len = 20
            input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            # Only first 15 tokens are real; rest are padding
            attention_mask[:, 15:] = 0
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    processor.side_effect = fake_call
    processor.__call__ = fake_call
    return processor


class TestVLMCollator:
    def test_output_has_required_keys(self, mock_processor: MagicMock) -> None:
        collator = VLMCollator(mock_processor)
        image = Image.new("RGB", (224, 224))
        batch = [(image, "What number?", "23")]
        result = collator(batch)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_labels_have_masked_prompt_tokens(self, mock_processor: MagicMock) -> None:
        collator = VLMCollator(mock_processor)
        image = Image.new("RGB", (224, 224))
        batch = [(image, "What number?", "23")]
        result = collator(batch)
        labels = result["labels"]
        # Some tokens should be masked to -100 (prompt portion)
        assert (labels == VLMCollator.IGNORE_INDEX).any()

    def test_labels_have_some_non_masked_tokens(
        self, mock_processor: MagicMock
    ) -> None:
        """At least some tokens should contribute to loss (the answer)."""
        collator = VLMCollator(mock_processor)
        image = Image.new("RGB", (224, 224))
        batch = [(image, "What number?", "23")]
        result = collator(batch)
        labels = result["labels"]
        # Not all tokens should be masked
        assert (labels != VLMCollator.IGNORE_INDEX).any()

    def test_batch_size_matches_input(self, mock_processor: MagicMock) -> None:
        collator = VLMCollator(mock_processor)
        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        batch = [(img, "What number?", str(i)) for i, img in enumerate(images)]
        result = collator(batch)
        assert result["input_ids"].shape[0] == 3
        assert result["labels"].shape[0] == 3

    def test_padding_tokens_masked_in_labels(self, mock_processor: MagicMock) -> None:
        """Padding tokens (attention_mask=0) should be masked to -100 in labels."""

        # Override processor to return some padding
        def fake_call_with_padding(**kwargs: object) -> dict[str, torch.Tensor]:
            input_ids = torch.ones(1, 20, dtype=torch.long)
            attention_mask = torch.ones(1, 20, dtype=torch.long)
            attention_mask[0, 15:] = 0  # last 5 tokens are padding
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

        mock_processor.__call__ = fake_call_with_padding
        mock_processor.side_effect = fake_call_with_padding

        collator = VLMCollator(mock_processor)
        image = Image.new("RGB", (224, 224))
        batch = [(image, "What number?", "23")]
        result = collator(batch)
        labels = result["labels"]
        # Padding positions should be -100
        assert (labels[0, 15:] == VLMCollator.IGNORE_INDEX).all()

    def test_max_length_passed_to_processor(self, mock_processor: MagicMock) -> None:
        collator = VLMCollator(mock_processor, max_length=128)
        assert collator.max_length == 128
