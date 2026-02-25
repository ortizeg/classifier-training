"""SmolVLM2 with LoRA for jersey number classification.

NOT extending BaseClassificationModel — uses causal LM loss (text generation),
not logits-based cross-entropy.
"""

from __future__ import annotations

import os
from typing import Any

# Must be set before any CUDA operations to reduce memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import lightning as L
import torch
from loguru import logger

from classifier_training.inference.vlm_inferencer import _parse_vlm_response
from classifier_training.utils.hydra import register


@register(name="smolvlm2", group="model")
class SmolVLM2ClassificationModel(L.LightningModule):
    """SmolVLM2 with LoRA for jersey number classification.

    Uses causal LM loss from the HuggingFace model, not cross-entropy on logits.
    LoRA adapters are applied to attention projection layers. The vision encoder
    connector is unfrozen to allow vision-language alignment.

    Args:
        model_name: HuggingFace model identifier.
        num_classes: Number of output classes (for compatibility; not used in loss).
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        max_new_tokens: Maximum tokens to generate during validation.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        num_classes: int = 43,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_new_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._num_classes = num_classes
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens

        # These will be set by the datamodule via train.py or manually
        self._class_to_idx: dict[str, int] = {}
        self._idx_to_class: dict[int, str] = {}
        self._processor: Any = None

        # Validation tracking
        self._val_correct = 0
        self._val_total = 0

        # Lazy-load model — defer to setup() or first forward pass
        self._model: Any = None
        self._lora_r = lora_r
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout

    @property
    def num_classes(self) -> int:
        """Number of output classes (for compatibility)."""
        return self._num_classes

    def set_class_weights(self, weights: torch.Tensor) -> None:
        """No-op. VLM uses causal LM loss, not weighted cross-entropy."""

    def set_class_mappings(
        self, class_to_idx: dict[str, int], idx_to_class: dict[int, str]
    ) -> None:
        """Set class mappings for validation response parsing."""
        self._class_to_idx = class_to_idx
        self._idx_to_class = idx_to_class

    def setup(self, stage: str | None = None) -> None:
        """Load model, apply LoRA, and set up freeze strategy."""
        if self._model is not None:
            return

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        # Determine dtype and attention implementation
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            dtype = torch.bfloat16
            try:
                import flash_attn  # type: ignore[import-not-found]  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                # SDPA (torch.nn.functional.scaled_dot_product_attention) provides
                # flash-attention-like O(n) memory for long sequences without the
                # flash_attn package. Critical for SmolVLM2 which generates ~1377
                # image tokens per sample.
                attn_impl = "sdpa"
                logger.info("flash_attn not installed, using SDPA attention")
        else:
            dtype = torch.float32
            attn_impl = "eager"

        self._processor = AutoProcessor.from_pretrained(self._model_name)  # type: ignore[no-untyped-call]

        # QLoRA: load model in 4-bit to fit on L4 GPU (24GB)
        if use_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(
                f"Loading VLM {self._model_name} in 4-bit QLoRA "
                f"(attn={attn_impl}, compute_dtype={dtype})"
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_name,
                quantization_config=bnb_config,
                _attn_implementation=attn_impl,
            )
            self._model = prepare_model_for_kbit_training(
                self._model, use_gradient_checkpointing=True
            )
        else:
            logger.info(
                f"Loading VLM {self._model_name} (attn={attn_impl}, dtype={dtype})"
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
                _attn_implementation=attn_impl,
            )
            self._model.gradient_checkpointing_enable()

        # Apply LoRA
        lora_config = LoraConfig(
            r=self._lora_r,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)

        # Log trainable parameters
        trainable, total = self._model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

    def forward(self, **kwargs: Any) -> Any:
        """Forward pass through the model."""
        return self._model(**kwargs)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Forward with labels — model computes causal LM loss internally."""
        outputs = self._model(**batch)
        loss: torch.Tensor = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Generate responses and compute accuracy against ground truth."""
        # Compute validation loss
        outputs = self._model(**batch)
        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True)

        # Generate predictions for accuracy
        if not self._class_to_idx or self._processor is None:
            return

        # Extract only the inputs needed for generation (no labels)
        gen_inputs = {k: v for k, v in batch.items() if k not in ("labels",)}

        with torch.no_grad():
            output_ids = self._model.generate(
                **gen_inputs, max_new_tokens=self._max_new_tokens
            )

        # Decode and parse each response
        prompt_len = batch["input_ids"].shape[-1]
        labels = batch["labels"]

        for i in range(output_ids.shape[0]):
            # Decode generated text (after prompt)
            generated = output_ids[i][prompt_len:]
            text = self._processor.decode(generated, skip_special_tokens=True)

            # Parse prediction
            preds = _parse_vlm_response(text, self._class_to_idx, self._idx_to_class)

            # Find ground truth from labels (first non-masked token)
            gt_tokens = labels[i][labels[i] != -100]
            if len(gt_tokens) == 0:
                continue
            gt_text = self._processor.decode(gt_tokens, skip_special_tokens=True)
            gt_text = gt_text.strip()

            # Compare
            if preds and preds[0].label == gt_text:
                self._val_correct += 1
            self._val_total += 1

    def on_validation_epoch_end(self) -> None:
        """Log validation accuracy."""
        if self._val_total > 0:
            accuracy = self._val_correct / self._val_total
            self.log("val/accuracy", accuracy, prog_bar=True)
        self._val_correct = 0
        self._val_total = 0

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """AdamW with cosine LR schedule (no warmup for LoRA)."""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
        max_epochs = (self.trainer.max_epochs or 10) if self.trainer else 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs),
            eta_min=self.hparams["learning_rate"] * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
