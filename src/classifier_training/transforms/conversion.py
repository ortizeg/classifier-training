"""Data conversion transforms for classification images."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.transforms import v2


class ToFloat32Tensor(v2.Transform):
    """Convert PIL images to float32 tensors.

    Wraps ``v2.ToImage`` + ``v2.ToDtype`` into a single transform that can be
    expressed in Hydra YAML (``torch.float32`` is not a valid YAML value).

    Args:
        scale: If ``True``, scale pixel values from ``[0, 255]`` to
            ``[0.0, 1.0]``.  If ``False`` (default), keep 0-255 range
            as float32.
    """

    def __init__(self, scale: bool = True) -> None:
        super().__init__()
        self._to_image = v2.ToImage()
        self._to_dtype = v2.ToDtype(torch.float32, scale=scale)

    def forward(self, *inputs: Any) -> Any:
        outputs = self._to_image(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return self._to_dtype(*outputs)
