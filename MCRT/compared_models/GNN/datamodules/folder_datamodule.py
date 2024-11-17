from typing import Dict, List, Any, Optional

from pymatgen.core import Structure

import functools
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from torch.utils.data import DataLoader, random_split

from GNN.datasets.folder_dataset import FolderDataset
from GNN.datamodules.base_datamodule import BaseDataModule


class FolderDataModule(BaseDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.seed = _config["split_seed"]
        self.data_dir = _config["data_dir"]
        
        self.downstream = _config["target"]
        self.compute_line_graph = _config["compute_line_graph"]
        self.neighbor_strategy = _config["neighbor_strategy"]
        self.cutoff = _config["cutoff"]
        self.max_neighbors = _config["max_neighbors"]
        self.use_canonize = _config["use_canonize"]
        self.train_ratio = _config["train_ratio"]
        self.val_ratio = _config["val_ratio"]
        self.test_ratio = _config["test_ratio"]
        self.num_workers = _config["num_workers"]


    @property
    def dataset_cls(self) -> DGLDataset:
        return FolderDataset

    @property
    def dataset_name(self) -> str:
        return "folder"
    
    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(  
                data_dir=self.data_dir,
                split="train",
                downstream = self.downstream,
                compute_line_graph=self.compute_line_graph,
                neighbor_strategy=self.neighbor_strategy,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_canonize=self.use_canonize,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(  
                data_dir=self.data_dir,
                split="val",
                downstream = self.downstream,
                compute_line_graph=self.compute_line_graph,
                neighbor_strategy=self.neighbor_strategy,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_canonize=self.use_canonize,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(  
                data_dir=self.data_dir,
                split="test",
                downstream = self.downstream,
                compute_line_graph=self.compute_line_graph,
                neighbor_strategy=self.neighbor_strategy,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_canonize=self.use_canonize,
        )
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.set_train_dataset()
            self.set_val_dataset()

        if stage in (None, "test"):
            self.set_test_dataset()

        self.collate = functools.partial(
            self.dataset_cls.collate_fn,
        )

    def _set_dataloader(
        self,
        dataset: DGLDataset,
        shuffle: bool,
        persistent_workers=False

    ) -> GraphDataLoader:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate,
            # use_ddp=self.use_ddp,
            drop_last=False,
            persistent_workers=persistent_workers
        )

    def train_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.train_dataset, shuffle=True,persistent_workers=True,)

    def val_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.val_dataset, shuffle=False,persistent_workers=True,)

    def test_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.test_dataset, shuffle=False)
