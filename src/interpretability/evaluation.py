import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
from pytorch_lightning.callbacks import Callback


class MetricsVisualizationCallback(Callback):
    def __init__(self, class_names=None):
        super().__init__()
        self.class_names = (
            class_names if class_names is not None else [str(i) for i in range(14)]
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get predictions and targets from validation set
        val_predictions = []
        val_targets = []

        for batch in trainer.val_dataloaders:
            x, y = batch
            with torch.no_grad():
                y_hat = pl_module(x.to(pl_module.device))
            val_predictions.extend(y_hat.cpu().numpy())
            val_targets.extend(y.cpu().numpy())

        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)

        # Create confusion matrix
        y_pred = (val_predictions > 0.5).astype(int)
        for i in range(14):
            cm = confusion_matrix(val_targets[:, i], y_pred[:, i])

            # Plot and log confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            plt.title(f"Confusion Matrix - {self.class_names[i]}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

            # Log to TensorBoard
            trainer.logger.experiment.add_figure(
                f"confusion_matrix/{self.class_names[i]}",
                plt.gcf(),
                global_step=trainer.global_step,
            )
            plt.close()

        # Calculate and log classification report
        report = classification_report(
            val_targets, y_pred, target_names=self.class_names, output_dict=True
        )

        # Log detailed metrics to TensorBoard
        for class_name in self.class_names:
            metrics = report[class_name]
            trainer.logger.experiment.add_scalars(
                f"detailed_metrics/{class_name}",
                {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                },
                global_step=trainer.global_step,
            )
