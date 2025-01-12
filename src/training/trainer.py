# src/training/train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import structlog

from src.callbacks.confusion_matrix import ConfusionMatrixLogger
from src.models.chestnets import ChestNetS
from src.models.lightning_module import ChestXRayModule
from src.data.datamodule import ChestDataModule
from src.configs.config import TrainingConfig, PathConfig

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def main():
    # Instantiate configs
    train_config = TrainingConfig(num_epochs=50)
    path_config = PathConfig()

    # Prepare device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Using device", device=device)

    # DataModule
    data_module = ChestDataModule(train_config, path_config)

    # Model
    model = ChestNetS()
    lightning_model = ChestXRayModule(
        model=model,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        num_classes=14,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="chest-xray-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=train_config.patience, mode="min"
    )

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir="logs", name="chest_xray", default_hp_metric=False
    )

    confusion_matrix_logger = ConfusionMatrixLogger()

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config.num_epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping, confusion_matrix_logger],
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train
    trainer.fit(lightning_model, data_module)

    # Test
    trainer.test(lightning_model, data_module)


if __name__ == "__main__":
    main()
