import seaborn as sns
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import confusion_matrix


class ConfusionMatrixLogger(Callback):
    def __init__(self, class_names=None):
        super().__init__()
        self.class_names = class_names or [f"Class {i}" for i in range(14)]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get predictions and targets from validation set
        val_loader = trainer.datamodule.val_dataloader()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(pl_module.device)
                y_hat = pl_module(x)

                pred_class = torch.argmax(y_hat, dim=1)
                target_class = torch.argmax(y, dim=1)

                predictions.extend(pred_class.cpu().numpy())
                targets.extend(target_class.cpu().numpy())

        # Create confusion matrix
        cm = confusion_matrix(targets, predictions)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Log to tensorboard
        trainer.logger.experiment.add_figure(
            "confusion_matrix", plt.gcf(), global_step=trainer.global_step
        )
        plt.close()
