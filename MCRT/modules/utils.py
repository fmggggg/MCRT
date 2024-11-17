import sys
import warnings
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)
from MCRT.models import DEFAULT_MCRT_PATH
from MCRT.modules.metrics import Accuracy, Scalar
if pl.__version__ >= '2.0.0':
    from pytorch_lightning.trainer.connectors.accelerator_connector import _AcceleratorConnector as AC
else:
    from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector as AC


_IS_INTERACTIVE = hasattr(sys, "ps1")


class ConfigurationError(Exception):
    pass

def _set_loss_names(loss_name):
    if isinstance(loss_name, list):
        d = {k: 1 for k in loss_name}
    elif isinstance(loss_name, str):
        d = {loss_name: 1}
    elif isinstance(loss_name, dict):
        d = loss_name
    elif loss_name is None:
        d = {}
    else:
        raise ConfigurationError(
            f"loss_name must be list, str, or dict, not {type(loss_name)}"
        )
    return _loss_names(d)


def _loss_names(d):
    ret = {
        "map": 0,  # Masked Atom Prediction
        "apc": 0,  # Atom pair classification  
        "sgp": 0,  # Space group prediction
        "sep": 0,  # symmetry element prediction
        "cdp": 0,  # Crystal density prediction
        "adp": 0,  # atom distance prediction
        "aap": 0,  # atom angle prediction
        "ucp": 0,  # unit cell prediction
        "classification": 0,  # classification
        "regression": 0,  # regression
    }
    ret.update(d)
    return ret

def _set_load_path(path):
    if path == 'MCRT':
        return DEFAULT_MCRT_PATH
    elif not path:
        return ""
    elif str(path)[-4:] == 'ckpt':
        return path
    else:
        raise ConfigurationError(
            f"path must be 'MCRT', None, or *.ckpt, not {path}"
        )

def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0
    ]
    return

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v <= 0:
                continue
            if k == "regression" or k == "cdp" or k == "adp"  or k == "aap" or k == "ucp":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def epoch_wrapup(pl_module):
    """
    compute loss and acc/mae for each epoch and reset metrics, and compute total metric by combining mae/acc
    """
    phase = "train" if pl_module.training else "val"

    the_metric = 0

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v <= 0:
            continue

        if loss_name == "regression" or loss_name == "cdp" or loss_name == "adp" or loss_name == "aap" or loss_name == "ucp":
            # mse loss
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            # mae loss
            value = getattr(pl_module, f"{phase}_{loss_name}_mae").compute()
            pl_module.log(
                f"{loss_name}/{phase}/mae_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_mae").reset()

            value = -value
        else:
            # classification acc
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(
                f"{loss_name}/{phase}/accuracy_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            # classification loss
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["regression_head", "classification_head"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps == -1:
        max_steps = pl_module.trainer.estimated_stepping_batches
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {pl_module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )

def get_num_devices(_config):
    if isinstance(devices := _config["devices"], list):
        devices = len(devices)
    elif isinstance(devices, int):
        pass
    elif devices == "auto" or devices is None:
        devices = _get_auto_device(_config)
    else:
        raise ConfigurationError(
            f'devices must be int, list, and "auto", not {devices}'
        )
    return devices

def _get_auto_device(_config):
    accelerator = AC(accelerator=_config["accelerator"]).accelerator
    devices = accelerator.auto_device_count()
    
    return devices

def _set_valid_batchsize(_config, devices):
    per_gpu_batchsize = _config["batch_size"] // devices

    _config["per_gpu_batchsize"] = per_gpu_batchsize
    warnings.warn(
        "'Per_gpu_batchsize' is larger than 'batch_size'.\n"
        f" Adjusted to per_gpu_batchsize to {per_gpu_batchsize}"
    )

def _check_valid_num_gpus(_config):
    devices = get_num_devices(_config)

    if devices > _config["batch_size"]:
        raise ConfigurationError(
            "Number of devices must be smaller than batch_size. "
            f'num_gpus : {devices}, batch_size : {_config["batch_size"]}'
        )

    if _IS_INTERACTIVE and devices > 1:
        _config["devices"] = 1
        warnings.warn(
            "The interactive environment (ex. jupyter notebook) does not supports multi-devices environment. "
            f"Adjusted number of devices : {devices} to 1. "
            "If you want to use multi-devices, make *.py file and run."
        )
    
    return devices

def get_valid_config(_config):
    # set loss_name to dictionary
    _config["loss_names"] = _set_loss_names(_config["loss_names"])

    # set load_path to directory
    _config["load_path"] = _set_load_path(_config["load_path"])

    # check_valid_num_gpus
    devices = _check_valid_num_gpus(_config)

    # Batch size must be larger than gpu_per_batch
    if _config["batch_size"] < _config["per_gpu_batchsize"] * devices:
        _set_valid_batchsize(_config, devices)

    return _config

class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean, std):
        if mean and std:
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

        self.mean = mean
        self.std = std

    def encode(self, tensor):
        return self._norm_func(tensor)

    def decode(self, tensor):
        return self._denorm_func(tensor)