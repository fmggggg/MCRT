import copy

import pytorch_lightning as pl

from GNN.config import ex
from GNN.datamodules import _datamodules
from GNN.models import _models
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import mean_absolute_error, r2_score



class PredictionCollector(Callback):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.targets = []
        self.cif_ids = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        preds = outputs["logits"]
        labels = outputs["target"]
        cif_ids = outputs["cif_id"]

        self.predictions.extend(preds.cpu().detach().numpy())
        self.targets.extend(labels.cpu().detach().numpy())
        self.cif_ids.extend(cif_ids)

    def on_test_end(self, trainer, pl_module):
        preds_array = np.array(self.predictions)
        targets_array = np.array(self.targets)

        mae = mean_absolute_error(targets_array, preds_array)
        r2 = r2_score(targets_array, preds_array)

        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")

        df = pd.DataFrame({
            "CIF_ID": self.cif_ids,
            "Predictions": preds_array.flatten(),
            "True Labels": targets_array.flatten()
        })
        df.to_csv("GNN_predictions_and_labels.csv", index=False)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    exp_name = _config["exp_name"]
    # set datamodule
    dm = _datamodules[_config["source"]](_config)
    # prepare data
    dm.prepare_data()
    # set model
    model = _models[_config["model_name"]](_config)
    print(model)
    # set checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/loss",
        mode="min",
        filename="best-{epoch}",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    prediction_collector = PredictionCollector()
    callbacks = [checkpoint_callback, lr_callback, prediction_collector]
    # set logger
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f"{exp_name}",
    )
    # gradient accumulation
    accumulate_grad_batches = _config["batch_size"] // (_config["per_gpu_batchsize"])
        
    # set trainer
    trainer = pl.Trainer(
        devices=_config["devices"],
        accelerator=_config["accelerator"],
        max_epochs=_config["max_epochs"],
        # strategy="ddp_find_unused_parameters_true",
        deterministic=_config["deterministic"],
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
    )

    
    if not _config["test_only"]:
        trainer.fit(model, dm, ckpt_path=_config["resume_from"])
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        model = _models[_config["model_name"]].load_from_checkpoint(_config["load_path"], strict=False)
        print(f"load model from {_config['load_path']}")
        trainer.test(model, datamodule=dm)


