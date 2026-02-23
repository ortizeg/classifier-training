"""Unit tests for annotation parsing functions in scripts/annotate_metadata.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts to path so we can import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from annotate_metadata import (
    _normalize_color,
    parse_border_response,
    parse_color_response,
    parse_single_response,
)


class TestParseSingleResponse:
    def test_valid_json(self) -> None:
        text = '{"jersey_color": "white", "number_color": "red", "border": true}'
        result = parse_single_response(text)
        assert result is not None
        assert result["jersey_color"] == "white"
        assert result["number_color"] == "red"
        assert result["border"] is True

    def test_json_with_surrounding_text(self) -> None:
        text = (
            'Here is the answer: {"jersey_color": "blue", '
            '"number_color": "white", "border": false} done.'
        )
        result = parse_single_response(text)
        assert result is not None
        assert result["jersey_color"] == "blue"
        assert result["number_color"] == "white"
        assert result["border"] is False

    def test_invalid_color_becomes_unknown(self) -> None:
        text = '{"jersey_color": "lime", "number_color": "white", "border": false}'
        result = parse_single_response(text)
        assert result is not None
        assert result["jersey_color"] == "unknown"

    def test_completely_invalid_text_returns_none(self) -> None:
        result = parse_single_response("I cannot determine the colors.")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = parse_single_response("")
        assert result is None

    def test_border_as_string_yes(self) -> None:
        text = '{"jersey_color": "white", "number_color": "black", "border": "yes"}'
        result = parse_single_response(text)
        assert result is not None
        assert result["border"] is True

    def test_border_as_string_no(self) -> None:
        text = '{"jersey_color": "white", "number_color": "black", "border": "no"}'
        result = parse_single_response(text)
        assert result is not None
        assert result["border"] is False

    def test_all_valid_colors_accepted(self) -> None:
        valid = [
            "white", "black", "red", "blue", "navy", "yellow",
            "green", "purple", "orange", "grey", "maroon", "teal", "pink",
        ]
        for color in valid:
            text = (
                f'{{"jersey_color": "{color}", '
                f'"number_color": "{color}", "border": false}}'
            )
            result = parse_single_response(text)
            assert result is not None
            assert result["jersey_color"] == color


class TestNormalizeColor:
    def test_valid_color(self) -> None:
        assert _normalize_color("red") == "red"

    def test_uppercase(self) -> None:
        assert _normalize_color("RED") == "red"

    def test_whitespace(self) -> None:
        assert _normalize_color("  blue  ") == "blue"

    def test_invalid_color(self) -> None:
        assert _normalize_color("magenta") == "unknown"

    def test_empty_string(self) -> None:
        assert _normalize_color("") == "unknown"


class TestParseColorResponse:
    def test_simple_color(self) -> None:
        assert parse_color_response("white") == "white"

    def test_with_whitespace_and_caps(self) -> None:
        assert parse_color_response("  Navy  ") == "navy"

    def test_invalid(self) -> None:
        assert parse_color_response("rainbow") == "unknown"


class TestParseBorderResponse:
    @pytest.mark.parametrize("text", ["yes", "Yes", "YES", "true", "True", "1"])
    def test_positive(self, text: str) -> None:
        assert parse_border_response(text) is True

    @pytest.mark.parametrize("text", ["no", "No", "false", "False", "0", "nope"])
    def test_negative(self, text: str) -> None:
        assert parse_border_response(text) is False
