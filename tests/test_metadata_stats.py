"""Unit tests for metadata statistics aggregation in scripts/metadata_stats.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from metadata_stats import collect_stats


class TestCollectStats:
    def test_basic_counts(self) -> None:
        records = [
            {
                "suffix": "0",
                "metadata": {
                    "jersey_color": "white",
                    "number_color": "red",
                    "border": True,
                },
            },
            {
                "suffix": "1",
                "metadata": {
                    "jersey_color": "blue",
                    "number_color": "white",
                    "border": False,
                },
            },
        ]
        (
            jersey_counts,
            number_counts,
            border_counts,
            _jbc, _nbc, _bbc,
            total_with,
            total_without,
            _coverage,
        ) = collect_stats(records)

        assert jersey_counts["white"] == 1
        assert jersey_counts["blue"] == 1
        assert number_counts["red"] == 1
        assert number_counts["white"] == 1
        assert border_counts[True] == 1
        assert border_counts[False] == 1
        assert total_with == 2
        assert total_without == 0

    def test_records_without_metadata(self) -> None:
        records = [
            {"suffix": "0"},
            {
                "suffix": "0",
                "metadata": {
                    "jersey_color": "white",
                    "number_color": "black",
                    "border": False,
                },
            },
        ]
        (
            _jc, _nc, _bc,
            _jbc, _nbc, _bbc,
            total_with, total_without,
            coverage_by_class,
        ) = collect_stats(records)

        assert total_with == 1
        assert total_without == 1
        assert coverage_by_class["0"] == (1, 1)

    def test_per_class_breakdown(self) -> None:
        records = [
            {
                "suffix": "5",
                "metadata": {
                    "jersey_color": "white",
                    "number_color": "blue",
                    "border": True,
                },
            },
            {
                "suffix": "5",
                "metadata": {
                    "jersey_color": "red",
                    "number_color": "white",
                    "border": False,
                },
            },
            {
                "suffix": "10",
                "metadata": {
                    "jersey_color": "white",
                    "number_color": "black",
                    "border": True,
                },
            },
        ]
        (
            _jc, _nc, _bc,
            jersey_by_class, _nbc, border_by_class,
            *_rest,
        ) = collect_stats(records)

        assert jersey_by_class["5"]["white"] == 1
        assert jersey_by_class["5"]["red"] == 1
        assert jersey_by_class["10"]["white"] == 1
        assert border_by_class["5"][True] == 1
        assert border_by_class["5"][False] == 1
        assert border_by_class["10"][True] == 1

    def test_empty_records(self) -> None:
        (
            jersey_counts,
            _nc, _bc,
            _jbc, _nbc, _bbc,
            total_with, total_without,
            coverage_by_class,
        ) = collect_stats([])

        assert len(jersey_counts) == 0
        assert total_with == 0
        assert total_without == 0
        assert len(coverage_by_class) == 0

    def test_border_string_parsing(self) -> None:
        """Border stored as string 'true'/'false' should be parsed correctly."""
        records = [
            {
                "suffix": "0",
                "metadata": {
                    "jersey_color": "white",
                    "number_color": "black",
                    "border": "true",
                },
            },
        ]
        (
            _jc, _nc, border_counts,
            *_rest,
        ) = collect_stats(records)

        assert border_counts[True] == 1
