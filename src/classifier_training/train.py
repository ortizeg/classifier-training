"""Training entrypoint for classifier_training.

Usage:
    pixi run train                              # defaults
    pixi run train model=resnet50               # override model
    pixi run train data.batch_size=32           # override batch size
    pixi run train trainer.max_epochs=10        # override epochs
"""

import shutil
import sys
from pathlib import Path
from typing import Any

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# CRITICAL: import models to trigger @register decorators BEFORE Hydra parses config
import classifier_training.models  # noqa: F401


@hydra.main(
    version_base=None, config_path="conf", config_name="train_basketball_resnet18"
)
def main(cfg: DictConfig) -> None:
    """Run training with the given Hydra config."""
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level=cfg.get("log_level", "INFO"))

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Seed everything for reproducibility
    L.seed_everything(cfg.get("seed", 42), workers=True)

    # Instantiate transforms (if configured)
    train_transforms = None
    val_transforms = None
    test_transforms = None
    if cfg.get("transforms"):
        tfm = cfg.transforms
        if tfm.get("train_transforms"):
            train_transforms = hydra.utils.instantiate(tfm.train_transforms)
        if tfm.get("val_transforms"):
            val_transforms = hydra.utils.instantiate(tfm.val_transforms)
        if tfm.get("test_transforms"):
            test_transforms = hydra.utils.instantiate(tfm.test_transforms)

    # Instantiate data module
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.data,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=test_transforms,
    )
    datamodule.setup("fit")

    # Instantiate model and set class weights from data
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    class_weights = datamodule.get_class_weights()  # type: ignore[attr-defined]
    model.set_class_weights(class_weights)  # type: ignore[operator]

    # Pass class mappings to VLM models for validation response parsing
    if hasattr(model, "set_class_mappings"):
        class_to_idx: dict[str, int] = datamodule.class_to_idx  # type: ignore[attr-defined]
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        model.set_class_mappings(class_to_idx, idx_to_class)  # type: ignore[operator]

    # Instantiate loggers
    loggers: list[Any] = []
    if cfg.get("logging"):
        for v in cfg.logging.values():
            if v is not None and "_target_" in v:
                loggers.append(hydra.utils.instantiate(v))

    # Instantiate callbacks (skip LearningRateMonitor when no logger is configured)
    callbacks: list[L.Callback] = []
    if cfg.get("callbacks"):
        for v in cfg.callbacks.values():
            if v is not None and "_target_" in v:
                if not loggers and "LearningRateMonitor" in v["_target_"]:
                    logger.warning("Skipping LearningRateMonitor: no logger configured")
                    continue
                callbacks.append(hydra.utils.instantiate(v))

    # Use original CWD for checkpoints during training (local FS is reliable).
    # GCS FUSE doesn't support PyTorch's temp-file-then-rename checkpoint pattern.
    local_root = HydraConfig.get().runtime.cwd
    trainer_cfg = dict(cfg.trainer)
    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=loggers or False,
        default_root_dir=local_root,
    )

    # Train! ckpt_path="last" enables automatic resume from last checkpoint
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")

    # Copy checkpoints to Hydra output dir (may be a GCS FUSE mount on Vertex AI).
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    ckpt_src = Path(local_root) / "checkpoints"
    if ckpt_src.is_dir() and output_dir != Path(local_root):
        ckpt_dst = output_dir / "checkpoints"
        logger.info(f"Copying checkpoints from {ckpt_src} to {ckpt_dst}")
        shutil.copytree(ckpt_src, ckpt_dst, dirs_exist_ok=True)
        logger.info(f"Checkpoint copy complete: {list(ckpt_dst.glob('*.ckpt'))}")


if __name__ == "__main__":
    main()
