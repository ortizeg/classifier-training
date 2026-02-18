"""Pydantic frozen configuration models for classifier_training."""

from pydantic import BaseModel, model_validator


class DataModuleConfig(BaseModel, frozen=True):
    """Configuration for ImageFolderDataModule.

    All fields are validated at construction time. Frozen — no mutation after creation.
    """

    data_root: str
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    image_size: int = 224

    @model_validator(mode="after")
    def _persistent_workers_requires_workers(self) -> "DataModuleConfig":
        """persistent_workers=True with num_workers=0 silently does nothing."""
        if self.persistent_workers and self.num_workers == 0:
            # Use object.__setattr__ because model is frozen
            object.__setattr__(self, "persistent_workers", False)
        return self
