# src/models/lightning_module.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)


class ChestXRayModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_classes: int = 14,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.criterion = nn.BCELoss()

        # Initialize metrics with proper settings
        metric_kwargs = {
            "num_classes": num_classes,
            "average": "macro",
        }

        self.train_accuracy = MulticlassAccuracy(**metric_kwargs)
        self.val_accuracy = MulticlassAccuracy(**metric_kwargs)
        self.test_accuracy = MulticlassAccuracy(**metric_kwargs)

        self.train_f1 = MulticlassF1Score(**metric_kwargs)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)

        self.train_auroc = MulticlassAUROC(**metric_kwargs)
        self.val_auroc = MulticlassAUROC(**metric_kwargs)
        self.test_auroc = MulticlassAUROC(**metric_kwargs)

        self.train_precision = MulticlassPrecision(**metric_kwargs)
        self.val_precision = MulticlassPrecision(**metric_kwargs)
        self.test_precision = MulticlassPrecision(**metric_kwargs)

        self.train_recall = MulticlassRecall(**metric_kwargs)
        self.val_recall = MulticlassRecall(**metric_kwargs)
        self.test_recall = MulticlassRecall(**metric_kwargs)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self(x)

        # Convert targets to float for BCE loss
        y_float = y.float()
        loss = self.criterion(y_hat, y_float)

        # Get metrics for current stage
        accuracy = getattr(self, f"{stage}_accuracy")
        f1 = getattr(self, f"{stage}_f1")
        auroc = getattr(self, f"{stage}_auroc")
        precision = getattr(self, f"{stage}_precision")
        recall = getattr(self, f"{stage}_recall")

        # Convert predictions to class indices for metrics
        preds_class = torch.argmax(y_hat, dim=1)
        target_class = torch.argmax(y, dim=1)

        # Update and log metrics
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"{stage}_accuracy", accuracy(preds_class, target_class), on_epoch=True
        )
        self.log(f"{stage}_f1", f1(preds_class, target_class), on_epoch=True)
        self.log(f"{stage}_auroc", auroc(y_hat, target_class), on_epoch=True)
        self.log(
            f"{stage}_precision", precision(preds_class, target_class), on_epoch=True
        )
        self.log(f"{stage}_recall", recall(preds_class, target_class), on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
