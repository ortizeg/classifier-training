"""Tests for Hydra config composition, overrides, and component instantiation.

Validates Phase 4 config wiring: all defaults resolve, trainer T4 settings,
data config, model/data instantiation, overrides, checkpoint paths, WandB config.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra.utils
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

# Trigger @register decorators before model instantiation tests
import classifier_training.models  # noqa: F401
from classifier_training.data.datamodule import ImageFolderDataModule
from classifier_training.models.resnet import (
    ResNet18ClassificationModel,
    ResNet50ClassificationModel,
)

CONF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "classifier_training", "conf")
)


@pytest.fixture()
def hydra_cfg() -> DictConfig:
    """Compose the root training config and yield it, clearing GlobalHydra after."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="train_basketball_resnet18")
        yield cfg
    GlobalHydra.instance().clear()


@pytest.fixture()
def hydra_cfg_with_overrides() -> callable:
    """Factory fixture for composing config with overrides."""

    def _compose(overrides: list[str]) -> DictConfig:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            return compose(config_name="train_basketball_resnet18", overrides=overrides)

    yield _compose
    GlobalHydra.instance().clear()


# --- Test 1: Config composition ---


def test_hydra_config_composes(hydra_cfg: DictConfig) -> None:
    """Root config loads without error; all 5 defaults resolve."""
    assert "model" in hydra_cfg
    assert "data" in hydra_cfg
    assert "trainer" in hydra_cfg
    assert "callbacks" in hydra_cfg
    assert "logging" in hydra_cfg


# --- Test 2: Trainer config values ---


def test_trainer_config_values(hydra_cfg: DictConfig) -> None:
    """Verify T4 defaults (TRAIN-02, TRAIN-03, TRAIN-04)."""
    assert hydra_cfg.trainer.precision == "16-mixed"  # TRAIN-02
    assert hydra_cfg.trainer.gradient_clip_val == 1.0  # TRAIN-03
    assert hydra_cfg.trainer.accumulate_grad_batches == 1  # TRAIN-03
    assert hydra_cfg.trainer.max_epochs == 200
    assert hydra_cfg.trainer.accelerator == "auto"


# --- Test 3: Data config values ---


def test_data_config_values(hydra_cfg: DictConfig) -> None:
    """Verify basketball dataset defaults (TRAIN-04, TRAIN-06)."""
    assert hydra_cfg.data.batch_size == 64  # TRAIN-04
    assert hydra_cfg.data.num_workers == 4  # TRAIN-04
    assert "basketball-jersey-numbers-ocr" in hydra_cfg.data.data_root  # TRAIN-06
    assert hydra_cfg.data.image_size == 224


# --- Test 4: Data config instantiation ---


def test_data_config_instantiates(hydra_cfg: DictConfig) -> None:
    """hydra.utils.instantiate(cfg.data) produces ImageFolderDataModule."""
    dm = hydra.utils.instantiate(hydra_cfg.data)
    assert isinstance(dm, ImageFolderDataModule)


# --- Test 5: Model config instantiation ---


def test_model_config_instantiates(hydra_cfg: DictConfig) -> None:
    """hydra.utils.instantiate(cfg.model) produces ResNet18."""
    model = hydra.utils.instantiate(hydra_cfg.model, pretrained=False)
    assert isinstance(model, ResNet18ClassificationModel)


# --- Test 6: Model override ---


def test_hydra_override_model(hydra_cfg_with_overrides: callable) -> None:
    """model=resnet50 override works (Phase 4 success criterion 5)."""
    cfg = hydra_cfg_with_overrides(["model=resnet50"])
    model = hydra.utils.instantiate(cfg.model, pretrained=False)
    assert isinstance(model, ResNet50ClassificationModel)


# --- Test 7: Batch size override ---


def test_hydra_override_batch_size(hydra_cfg_with_overrides: callable) -> None:
    """data.batch_size=32 override propagates (Phase 4 success criterion 5)."""
    cfg = hydra_cfg_with_overrides(["data.batch_size=32"])
    assert cfg.data.batch_size == 32
    dm = hydra.utils.instantiate(cfg.data)
    assert dm._config.batch_size == 32


# --- Test 8: Checkpoint resume path ---


def test_checkpoint_resume_path_resolves_stably(hydra_cfg: DictConfig) -> None:
    """ModelCheckpoint dirpath is fixed and train.py sets default_root_dir."""
    # 1. Verify config value is fixed (not timestamped)
    ckpt_cfg = hydra_cfg.callbacks.model_checkpoint
    assert ckpt_cfg.dirpath == "checkpoints"  # Fixed, not ${hydra:runtime.output_dir}
    assert "${" not in str(ckpt_cfg.dirpath)  # No Hydra interpolation

    # 2. Verify train.py sets default_root_dir (so "checkpoints" resolves stably)
    train_src = Path("src/classifier_training/train.py").read_text()
    assert "default_root_dir" in train_src, (
        "train.py must set default_root_dir for stable checkpoint resolution"
    )
    assert "HydraConfig.get().runtime.output_dir" in train_src, (
        "default_root_dir should use Hydra output dir for stable checkpoint resolution"
    )


# --- Test 9: WandB logger config ---


def test_wandb_logger_config(hydra_cfg: DictConfig) -> None:
    """WandbLogger config present in composed config."""
    assert "wandb" in hydra_cfg.logging
    assert hydra_cfg.logging.wandb._target_ == "lightning.pytorch.loggers.WandbLogger"
    assert hydra_cfg.logging.wandb.project == "classifier-training"


# --- Test 10: Seed and log level ---


def test_seed_and_log_level(hydra_cfg: DictConfig) -> None:
    """Global params from root config."""
    assert hydra_cfg.seed == 42
    assert hydra_cfg.log_level == "INFO"


# --- Test 11: WandB metric routing documented ---


def test_wandb_metric_routing_documented(hydra_cfg: DictConfig) -> None:
    """Verify WandbLogger config is correct for metric routing.

    NOTE: self.log() -> WandbLogger metric routing is handled by Lightning's
    built-in WandbLogger integration. When WandbLogger is passed to Trainer(logger=...),
    all self.log() calls in LightningModule automatically route to wandb.log().
    This is a Lightning framework guarantee, not custom code.

    This test verifies the config prerequisites are correct (logger target and project).
    """
    wandb_cfg = hydra_cfg.logging.wandb
    assert wandb_cfg._target_ == "lightning.pytorch.loggers.WandbLogger"
    # WandbLogger's __init__ accepts project, which Lightning uses for wandb.init()
    assert wandb_cfg.project == "classifier-training"
    # log_model=false prevents auto-uploading model artifacts
    assert wandb_cfg.log_model is False
