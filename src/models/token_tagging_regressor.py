import os
import inspect
from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric
from torchmetrics import MeanSquaredError
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
import numpy as np

from src.utils import utils
from src.utils.torch_utils import masked_loss
from src.utils.torch_metrics import MaskedMeanSquaredError


class TokenTaggingRegressor(LightningModule):
    """
    Transformer Model for Token Tagging, i.e. per Token Sequence Regression.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        huggingface_model: str,
        num_labels: int,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        mlp: nn.Module = None,
        p_dropout: float = 0.1,
        loss_fn: nn.Module = torch.nn.MSELoss(reduction="none"),
        save_path: str = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="loss_fn")

        # Load model and add classification head
        self.model = AutoModel.from_pretrained(huggingface_model)
        self.regressor = nn.Linear(self.model.config.hidden_size, num_labels)

        # Init classifier weights according to initialization rules of model
        self.model._init_weights(self.regressor)

        # Apply dropout rate of model
        dropout_prob = p_dropout  # self.model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        # loss function (assuming single-label multi-class classification)
        self.loss_fn = loss_fn

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MaskedMeanSquaredError()
        self.val_loss = MaskedMeanSquaredError()
        self.test_loss = MaskedMeanSquaredError()

        # for logging best so far validation accuracy
        self.val_loss_best = MinMetric()

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [
            param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        self.forward_signature = params

        # create save path dir
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(os.path.join(self.save_path, "predictions"), exist_ok=True)

    def forward(self, batch: Dict[str, torch.tensor]):
        # batch: dict with keys "input_ids", "attention_mask", "labels"
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        # print(f"outputs {outputs.shape}")
        outputs_dropout = self.dropout(outputs)
        logits = self.regressor(outputs_dropout)
        # squeeze to remove the last dimension (N_EMB -> 1) anyways
        logits = logits.squeeze(-1)
        # print(f"logits {logits.shape}")
        return logits

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()

    def step(self, batch: Dict[str, torch.tensor]):
        logits = self(batch)  # forward pass
        labels = batch["labels"]
        mask = batch["attention_mask"]  # ignore padded sequence in loss
        loss = masked_loss(
            labels=labels, predictions=logits, mask=mask, loss_fn=self.loss_fn
        )
        return loss, logits

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, train_loss tracks it over the epoch
        loss, preds = self.step(batch)
        self.train_loss(preds, batch["labels"], batch["attention_mask"])
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return {
            "loss": loss,
            "preds": preds,
            "targets": batch["labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, val_loss tracks it over the epoch
        loss, preds = self.step(batch)
        self.val_loss(preds, batch["labels"], batch["attention_mask"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "preds": preds,
            "targets": batch["labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, test_loss tracks it over the epoch
        loss, preds = self.step(batch)
        self.test_loss(preds, batch["labels"], batch["attention_mask"])
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # TODO: make callback work
        #
        # Save test data and predictions
        np.save(
            f"{self.save_path}/predictions/test_input_ids_{batch_idx}.npy",
            batch["input_ids"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_attention_mask_{batch_idx}.npy",
            batch["attention_mask"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_labels_{batch_idx}.npy",
            batch["labels"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_preds_{batch_idx}.npy",
            preds.cpu().numpy(),
        )
        #
        #
        #

        return {
            "loss": loss,
            "preds": preds,
            "targets": batch["labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_test_epoch_end(self):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()
        pass

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
