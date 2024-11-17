import functools
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from MCRT.data_processor.dataset import Dataset

class Datamodule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["root_dataset"]
        self.max_num_nbr = _config["max_num_nbr"]
        self.mask_probability = _config["mask_probability"]
        self.num_ap=_config["num_ap"]
        self.n_dist=_config["n_dist"]
        self.n_angle=_config["n_angle"]
        self.max_graph_len = _config["max_graph_len"]
        self.angle_nbr = _config["angle_nbr"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.downstream = _config["downstream"]
        self.nbr_fea_len = _config["nbr_fea_len"]
        self.if_conv = _config["if_conv"]
        self.pos_emb = _config["pos_emb"]
        self.if_alignn = _config["if_alignn"]
        self.if_image = _config["if_image"]
        self.if_grid = _config["if_grid"]
        self.read_from_pickle = _config["read_from_pickle"]
        self.tasks = [k for k, v in _config["loss_names"].items() if v > 0]

    @property
    def dataset_cls(self):
        return Dataset
    
    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(  
                data_dir=self.data_dir,
                split="train",
                max_num_nbr = self.max_num_nbr,
                nbr_fea_len=self.nbr_fea_len,
                num_ap=self.num_ap,
                n_dist=self.n_dist,
                n_angle=self.n_angle,
                downstream=self.downstream,
                tasks=self.tasks,
                if_conv=self.if_conv,
                pos_emb=self.pos_emb,
                if_alignn=self.if_alignn,
                if_image=self.if_image,
                if_grid=self.if_grid,
                read_from_pickle=self.read_from_pickle,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(  
                data_dir=self.data_dir,
                split="val",
                max_num_nbr = self.max_num_nbr,
                nbr_fea_len=self.nbr_fea_len,
                num_ap=self.num_ap,
                n_dist=self.n_dist,
                n_angle=self.n_angle,
                downstream=self.downstream,
                tasks=self.tasks,
                if_conv=self.if_conv,
                pos_emb=self.pos_emb,
                if_alignn=self.if_alignn,
                if_image=self.if_image,
                if_grid=self.if_grid,
                read_from_pickle=self.read_from_pickle,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(  
                data_dir=self.data_dir,
                split="test",
                max_num_nbr = self.max_num_nbr,
                nbr_fea_len=self.nbr_fea_len,
                num_ap=self.num_ap,
                n_dist=self.n_dist,
                n_angle=self.n_angle,
                downstream=self.downstream,
                tasks=self.tasks,
                if_conv=self.if_conv,
                pos_emb=self.pos_emb,
                if_alignn=self.if_alignn,
                if_image=self.if_image,
                if_grid=self.if_grid,
                read_from_pickle=self.read_from_pickle,
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()


        self.collate = functools.partial(
            self.dataset_cls.collate,
            max_graph_len=self.max_graph_len,
            angle_nbr=self.angle_nbr,
            mask_probability = self.mask_probability,
            tasks=self.tasks,
            if_conv=self.if_conv,
            pos_emb=self.pos_emb,
            if_alignn=self.if_alignn,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
