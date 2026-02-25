"""VLM dataset for jersey number classification fine-tuning."""

import json
from pathlib import Path

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from classifier_training.data.utils import get_files
from classifier_training.inference.vlm_inferencer import VLM_PROMPT


class VLMJerseyNumberDataset(Dataset[tuple[Image.Image, str, str]]):
    """Wraps JSONL annotations for VLM fine-tuning.

    Each ``__getitem__`` returns:
    - image: ``PIL.Image`` (RGB)
    - prompt: str (user prompt asking for jersey number)
    - answer: str (ground truth label from ``idx_to_class``)

    No torchvision transforms are applied — the HF processor handles
    image preprocessing during collation.

    Args:
        root: Split directory to search recursively for ``.jsonl`` files.
        class_to_idx: Alphabetically-ordered mapping from label string to int.
        prompt: User prompt template. Defaults to :data:`VLM_PROMPT`.
    """

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        prompt: str = VLM_PROMPT,
    ) -> None:
        self.root = root
        self.class_to_idx = class_to_idx
        self.idx_to_class: dict[int, str] = {v: k for k, v in class_to_idx.items()}
        self.prompt = prompt
        self.samples: list[tuple[Path, int]] = []

        ann_files = get_files(root, (".jsonl",))
        skipped = 0
        for ann_path in ann_files:
            ann_dir = ann_path.parent
            with open(ann_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    suffix = record["suffix"]
                    if suffix not in class_to_idx:
                        skipped += 1
                        continue
                    img_path = ann_dir / record["image"]
                    label_idx = class_to_idx[suffix]
                    self.samples.append((img_path, label_idx))
        if skipped:
            logger.warning(
                f"Skipped {skipped} annotation(s) with unknown labels under {root}"
            )
        logger.debug(
            f"VLMJerseyNumberDataset: loaded {len(self.samples)} samples "
            f"from {len(ann_files)} annotation file(s) under {root}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, str, str]:
        img_path, label_idx = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        answer = self.idx_to_class[label_idx]
        return img, self.prompt, answer
