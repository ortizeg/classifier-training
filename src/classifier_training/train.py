"""Training entrypoint for classifier_training.

Usage:
    pixi run train                              # defaults
    pixi run train model=resnet50               # override model
    pixi run train data.batch_size=32           # override batch size
    pixi run train trainer.max_epochs=10        # override epochs
"""

import sys
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

    # Instantiate callbacks
    callbacks: list[L.Callback] = []
    if cfg.get("callbacks"):
        for v in cfg.callbacks.values():
            if v is not None and "_target_" in v:
                callbacks.append(hydra.utils.instantiate(v))

    # Instantiate loggers
    loggers: list[Any] = []
    if cfg.get("logging"):
        for v in cfg.logging.values():
            if v is not None and "_target_" in v:
                loggers.append(hydra.utils.instantiate(v))

    # Build Trainer from config dict (NOT via hydra.utils.instantiate --
    # trainer config has no _target_ key)
    trainer_cfg = dict(cfg.trainer)
    trainer = L.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=loggers or False,
        default_root_dir=HydraConfig.get().runtime.cwd,
    )

    # Train! ckpt_path="last" enables automatic resume from last checkpoint
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")


if __name__ == "__main__":
    main()
