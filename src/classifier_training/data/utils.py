"""Utility functions for the data pipeline."""

from pathlib import Path


def get_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Recursively find files matching extensions under root.

    Args:
        root: Directory to search recursively.
        extensions: Tuple of lowercase extensions including dot
            (e.g., (".jpg", ".png")).

    Returns:
        Sorted list of matching file paths.
    """
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            files.append(p)
    return sorted(files)
