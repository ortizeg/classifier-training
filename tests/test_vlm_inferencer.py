"""Tests for SmolVLM2ClassificationInferencer response parsing."""

from __future__ import annotations

from classifier_training.inference.vlm_inferencer import _parse_vlm_response

# Shared class mappings for tests
CLASS_TO_IDX = {"": 0, "0": 1, "1": 2, "23": 3, "00": 4}
IDX_TO_CLASS = {0: "", 1: "0", 2: "1", 3: "23", 4: "00"}


class TestParseVlmResponse:
    def test_exact_match(self) -> None:
        preds = _parse_vlm_response("23", CLASS_TO_IDX, IDX_TO_CLASS)
        assert len(preds) == 1
        assert preds[0].label == "23"
        assert preds[0].class_id == 3
        assert preds[0].confidence == 1.0

    def test_with_whitespace(self) -> None:
        preds = _parse_vlm_response("  23  ", CLASS_TO_IDX, IDX_TO_CLASS)
        assert preds[0].label == "23"

    def test_with_quotes(self) -> None:
        preds = _parse_vlm_response('"23"', CLASS_TO_IDX, IDX_TO_CLASS)
        assert preds[0].label == "23"

    def test_with_period(self) -> None:
        preds = _parse_vlm_response("23.", CLASS_TO_IDX, IDX_TO_CLASS)
        assert preds[0].label == "23"

    def test_number_in_sentence(self) -> None:
        preds = _parse_vlm_response(
            "The number on the jersey is 23.", CLASS_TO_IDX, IDX_TO_CLASS
        )
        assert preds[0].label == "23"

    def test_unknown_number_falls_back(self) -> None:
        preds = _parse_vlm_response("99", CLASS_TO_IDX, IDX_TO_CLASS)
        # Falls back to empty-string class
        assert len(preds) == 1
        assert preds[0].label == ""
        assert preds[0].class_id == 0

    def test_no_number_falls_back(self) -> None:
        preds = _parse_vlm_response("I cannot see a number", CLASS_TO_IDX, IDX_TO_CLASS)
        assert len(preds) == 1
        assert preds[0].label == ""

    def test_single_digit(self) -> None:
        preds = _parse_vlm_response("1", CLASS_TO_IDX, IDX_TO_CLASS)
        assert preds[0].label == "1"
        assert preds[0].class_id == 2

    def test_zero(self) -> None:
        preds = _parse_vlm_response("0", CLASS_TO_IDX, IDX_TO_CLASS)
        assert preds[0].label == "0"
        assert preds[0].class_id == 1

    def test_double_zero(self) -> None:
        # "00" should be tried via direct match
        preds = _parse_vlm_response("00", CLASS_TO_IDX, IDX_TO_CLASS)
        assert preds[0].label == "00"
        assert preds[0].class_id == 4
