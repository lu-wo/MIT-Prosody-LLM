import inspect
from typing import Any, Dict, List, Tuple

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, MinMetric
from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

from src.utils import utils
from src.utils.torch_utils import masked_loss
from src.utils.torch_metrics import MaskedAccuracy


class TokenTaggingClassifier(LightningModule):
    def __init__(
        self,
        huggingface_model: str,
        num_labels: int,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        mlp: nn.Module = None,
        p_dropout: float = 0.1,
        loss_fn: nn.Module = torch.nn.CrossEntropyLoss(reduction="none"),
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = AutoModel.from_pretrained(huggingface_model)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

        self.model._init_weights(self.classifier)

        dropout_prob = p_dropout
        self.dropout = nn.Dropout(dropout_prob)

        self.loss_fn = loss_fn

        self.train_acc = MaskedAccuracy()
        self.val_acc = MaskedAccuracy()
        self.test_acc = MaskedAccuracy()

        self.val_acc_best = MinMetric()

        params = inspect.signature(self.model.forward).parameters.values()
        params = [
            param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        self.forward_signature = params

    def forward(self, batch: Dict[str, torch.tensor]):
        outputs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        outputs_dropout = self.dropout(outputs)
        logits = self.classifier(outputs_dropout)
        return logits

    def on_train_start(self):
        self.val_acc.reset()
        self.val_acc_best.reset()

    def step(self, batch: Dict[str, torch.tensor]):
        logits = self(batch)
        labels = batch["labels"]
        mask = batch["attention_mask"]
        loss = masked_loss(
            labels=labels, predictions=logits, mask=mask, loss_fn=self.loss_fn
        )
        return loss, logits

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)
        self.train_acc(preds, batch["labels"], batch["attention_mask"])
        self.log(
            "train/accuracy", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)
        self.val_acc(preds, batch["labels"], batch["attention_mask"])
        self.log(
            "val/accuracy", self.val_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/accuracy_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        loss, preds = self.step(batch)
        self.test_acc(preds, batch["labels"], batch["attention_mask"])
        self.log(
            "test/accuracy", self.test_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return {"loss": loss, "preds": preds, "targets": batch["labels"]}

    def on_test_epoch_end(self):
        pass

    def on_epoch_end(self):
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
        pass

    @property
    def total_training_steps(self) -> int:
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
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
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/accuracy",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
