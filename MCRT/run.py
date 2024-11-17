import sys
import os
import copy
import warnings
from pathlib import Path
import shutil
import torch

import pytorch_lightning as pl
from MCRT.config import ex
from MCRT.config import config as _config
from MCRT.modules.datamodule import Datamodule
from MCRT.modules.McrtModule import MCRT_Module
from MCRT.modules.utils import (
    get_valid_config,
    get_num_devices,
    ConfigurationError,
)

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"

    
)


_IS_INTERACTIVE = hasattr(sys, "ps1")

def run(root_dataset, downstream=None, log_dir="logs/", *, test_only=False, **kwargs):
    """
    Train or predict MCRT.

    Call signatures::
        run(root_dataset, downstream, [test_only], **kwargs)

    The basic usage of the code is as follows:

    >>> run(root_dataset, downstream)  # train MCRT from [root_dataset] with train_{downstream}.json
    >>> run(root_dataset, downstream, log_dir, test_only=True, load_path=model_path) # test MCRT from trained-model path


    Parameters
    __________
    :param root_dataset: 
    :param downstream: Name of user-specific task (e.g. bandgap, gasuptake, etc).
    :param log_dir: Directory to save log, models, and params.
    :param test_only: If True, only the test process is performed without the learning model.

    Other Parameters
    ________________
    load_path: str, default: None
    This parameter specifies the path of the model that will be used for training/testing.
    The available options are 'MCRT' (for finetune), None (for pretrain), or path to *.ckpt (for using finetuned model).

    loss_names: str or list, or dict, default: "regression"
        One or more of the following loss : 'regression', 'classification', 'map', 'apc', 'sgp' and 'cdp'

    n_classes: int, default: 0
        Number of classes when your loss is 'classification'

    batch_size: int, default: 32
        desired batch size; for gradient accumulation

    per_gpu_batchsize: int, default: 8
        you should define this manually with per_gpu_batch_size

    accelerator: str, default: 'auto'
        Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto")
        as well as custom accelerator instances.

    devices: int or list, default: "auto"
        Number of devices to train on (int), which devices to train on (list or str), or "auto".
        It will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type ("cpu", "gpu", "tpu", "ipu", "auto").

    num_nodes: int, default: 1
        Number of GPU nodes for distributed training.

    num_workers: int, default: 16
        the number of cpu's core

    precision: int or str, default: 16-mixed
        MCRT supports either double (64), float (32), bfloat16 (bf16), or half (16) precision training.
        Half precision, or mixed precision, is the combined use of 32 and 16 bit floating points to reduce memory footprint during model training.
        This can result in improved performance, achieving +3X speedups on modern GPUs.

    max_epochs: int, default: 20
        Stop training once this number of epochs is reached.

    seed: int, default: 0
        The random seed for pytorch_lightning.

    dataset_seed: int, default: 123
        The random seed for dataset split.


    Normalization parameters:
    _________________________
    mean: float or None, default: None
        mean for normalizer. If None, it is automatically obtained from the train dataset.

    std: float or None, default: None
        standard deviation for normalizer. If None, it is automatically obtained from the train dataset.


    Optimzer setting parameters
    ___________________________
    optim_type: str, default: "adamw"
        Type of optimizer, which is "adamw", "adam", or "sgd" (momentum=0.9)

    learning_rate: float, default: 1e-4
        Learning rate for optimizer

    weight_decay: float, default: 1e-2
        Weight decay for optmizer

    decay_power: float, default: 1
        default polynomial decay, [cosine, constant, constant_with_warmup]

    max_steps: int, default: -1
        num_data * max_epoch // batch_size (accumulate_grad_batches)
        if -1, set max_steps automatically.

    warmup_steps : int or float, default: 0.05
        warmup steps for optimizer. If type is float, set to max_steps * warmup_steps.

    end_lr: float, default: 0

    lr_mult: float, default: 1
        multiply lr for downstream heads


    Transformer setting parameters
    ______________________________
    hid_dim = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    mpp_ratio = 0.15


    Atom-based Graph Parameters
    ___________________________
    atom_fea_len = 64
    nbr_fea_len = 64
    max_graph_len = 300 # number of maximum nodes in graph
    max_nbr_atoms = 12


    Energy-grid Parameters
    ______________________
    img_size = 30
    patch_size = 5  # length of patch
    in_chans = 1  # channels of grid image
    max_grid_len = -1  # when -1, max_image_len is set to maximum ph*pw of batch images
    draw_false_grid = False


    Visuallization Parameters
    _________________________
    visualize: bool, default: False
        return attention map (use at attetion visualization step)


    Pytorch lightning setting parameters
    ____________________________________
    resume_from = None
    val_check_interval = 1.0
    dataset_size = False  # experiments for dataset size with 100 [k] or 500 [k]

    """

    config = copy.deepcopy(_config())
    for key in kwargs.keys():
        if key not in config:
            raise ConfigurationError(f"{key} is not in configuration.")

    config.update(kwargs)
    config["root_dataset"] = root_dataset
    config["downstream"] = downstream
    config["log_dir"] = log_dir
    config["test_only"] = test_only

    main(config)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    _config = get_valid_config(_config)
    dm = Datamodule(_config)
    model = MCRT_Module(_config)
    exp_name = f"{_config['exp_name']}"

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )

    if _config["test_only"]:
        name = f'test_{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'
    else:
        name = f'{exp_name}_seed{_config["seed"]}_from_{str(_config["load_path"]).split("/")[-1][:-5]}'

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=name,
    )
    # callbacks
    test_to_csv = _config["test_to_csv"]
    cls_to_csv = _config["cls_to_csv"]
    prediction_collector = PredictionCollector(test_to_csv=test_to_csv,cls_to_csv=cls_to_csv)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback,prediction_collector]

    num_device = get_num_devices(_config)
    print("num_device", num_device)

    # gradient accumulation
    if num_device == 0:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * _config["num_nodes"]
        )
    else:
        accumulate_grad_batches = _config["batch_size"] // (
            _config["per_gpu_batchsize"] * num_device * _config["num_nodes"]
        )

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    if _IS_INTERACTIVE:
        strategy = None
    else:
        strategy = _config["strategy"]
    torch.set_float32_matmul_precision('medium') 

    log_every_n_steps = 10

    trainer = pl.Trainer(
        accelerator=_config["accelerator"],
        devices=_config["devices"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        # strategy=strategy, 
        benchmark=True,
        max_epochs=_config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=_config["val_check_interval"],
        deterministic=True,
        gradient_clip_val=1.0,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        log_dir = Path(logger.log_dir)/'checkpoints'
        if best_model:= next(log_dir.glob('epoch=*.ckpt')):
            shutil.copy(best_model, log_dir/'best.ckpt')
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        trainer.test(model, datamodule=dm)


class PredictionCollector(Callback):
    def __init__(self, test_to_csv=False, cls_to_csv=False):
        super().__init__()
        self.predictions = []
        self.targets = []
        self.cif_ids = []
        self.cls_feats = []
        self.test_to_csv = test_to_csv
        self.cls_to_csv = cls_to_csv

        # For classification
        self.classification_preds = []
        self.classification_labels = []
        self.classification_cif_ids = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Regression predictions
        if 'regression_logits' in outputs.keys():
            preds = outputs["regression_logits"]
            labels = outputs["regression_labels"]

            self.predictions.extend(preds.cpu().detach().numpy())
            self.targets.extend(labels.cpu().detach().numpy())

        # Classification predictions
        if 'classification_logits' in outputs.keys():
            cls_preds = outputs["classification_logits"]
            cls_labels = outputs["classification_labels"]

            self.classification_preds.extend(cls_preds.cpu().detach().numpy())
            self.classification_labels.extend(cls_labels.cpu().detach().numpy())

        if 'cif_id' in outputs.keys():
            cif_ids = outputs['cif_id']
            self.cif_ids.extend(cif_ids)

            if 'classification_logits' in outputs.keys():
                self.classification_cif_ids.extend(cif_ids)

        if self.cls_to_csv:
            if 'cif_id' in outputs.keys() and 'cls_feats' in outputs.keys():
                cif_ids = outputs['cif_id']
                cls_feats = outputs['cls_feats']

                self.cif_ids.extend(cif_ids)
                self.cls_feats.extend(cls_feats.cpu().detach().numpy())

    def on_test_end(self, trainer, pl_module):
        # Regression results
        if self.predictions:
            preds_array = np.array(self.predictions)
            targets_array = np.array(self.targets)

            mae = mean_absolute_error(targets_array, preds_array)
            r2 = r2_score(targets_array, preds_array)

            print(f"Test MAE: {mae}")
            print(f"Test R2 Score: {r2}")

            if self.test_to_csv:
                # Save regression results to CSV
                df = pd.DataFrame({
                    "CIF_ID": self.cif_ids,
                    "Predictions": preds_array.flatten(),
                    "Labels": targets_array.flatten()
                })
                df.to_csv("test_predictions_and_labels.csv", index=False)

        # Classification results
        if self.classification_preds:
            cls_preds_array = np.argmax(np.array(self.classification_preds), axis=1)
            cls_labels_array = np.array(self.classification_labels)

            accuracy = accuracy_score(cls_labels_array, cls_preds_array)
            f1 = f1_score(cls_labels_array, cls_preds_array, average='weighted')

            print(f"Test Accuracy: {accuracy}")
            print(f"Test F1 Score: {f1}")

            if self.test_to_csv:
                # Save classification results to CSV
                df_cls = pd.DataFrame({
                    "CIF_ID": self.classification_cif_ids,
                    "Predictions": cls_preds_array,
                    "Labels": cls_labels_array
                })
                df_cls.to_csv("test_classification_predictions_and_labels.csv", index=False)

        # Save classification features if required
        if self.cls_to_csv and self.cif_ids:
            df_cls_feats = pd.DataFrame(self.cls_feats)
            df_cls_feats.insert(0, 'cif_id', self.cif_ids)
            df_cls_feats.columns = ['cif_id'] + [f'feat_{i}' for i in range(df_cls_feats.shape[1] - 1)]
            df_cls_feats.to_parquet("test_cls_feats.parquet", index=False)
            print("\nCLS parquet saved")
