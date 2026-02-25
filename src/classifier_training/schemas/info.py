"""Annotation metadata info schema."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class AnnotationInfo(BaseModel):
    """Source and context metadata for an annotation."""

    annotations_source: str
    image_width: int | None = None
    image_height: int | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
