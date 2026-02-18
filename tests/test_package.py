"""Smoke test: verify the classifier_training package is importable."""

import classifier_training


def test_package_version() -> None:
    """Package must declare a __version__ string."""
    assert isinstance(classifier_training.__version__, str)
    assert classifier_training.__version__ == "0.0.1"
