from typing import Dict, List, Any
from tqdm import tqdm
import os
from pymatgen.core import Structure
import csv
import dgl
import json
from dgl.data import DGLDataset
import torch
import pickle

from GNN.datasets.utils_jarvis import (
    jarvis_atoms_to_dgl_graph,
    compute_bond_cosines,
    convert_structures_to_jarvis_atoms,
)


class FolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        compute_line_graph: bool = True,
        downstream: str = "",
        neighbor_strategy: str = "k-nearest",
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        use_canonize: bool = True,
    ):
        """Generate folderDataset.

        Args:
            names (List[str]): a list of names
            structures (List[Structure]): a list of pymatgen Structure objects
            targets (List[Any]): a list of targets
            compute_line_graph (bool, optional): compute line graph. Defaults to False.
            neighbor_strategy (str, optional): neighbor strategy. Defaults to "k-nearest".
            cutoff (float, optional): cutoff distance. Defaults to 8.0.
            max_neighbors (int, optional): maximum number of neighbors. Defaults to 12.
            use_canonize (bool, optional): whether to use canonize. Defaults to True.
        """
        super().__init__()
        self.data_dir = data_dir
        assert split in {"train", "test", "val"}
        self.split = split
        self.compute_line_graph = compute_line_graph
        self.neighbor_strategy = neighbor_strategy
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.use_canonize = use_canonize

        downstream_file = os.path.join(self.data_dir, f"{downstream}.csv")
        self.id_downstream_data=self.get_prop(downstream_file)
        dataset_split = self.load_dataset_split()
        self.pickle_path=os.path.join(self.data_dir,"pickles") # pickles include the info for atom_label (for "map") and atm_label (for "apc")
        assert os.path.exists(self.pickle_path), 'pickles does not exist!'
        cif_ids_pickle=set([os.path.splitext(name)[0] for name in os.listdir(self.pickle_path) if name.endswith('.pickle')])
        self.cif_ids = [cid for cid in dataset_split[self.split] if cid in cif_ids_pickle]
        
        self.id_downstream_data=self.get_prop(downstream_file)
        downstream_cif_keys = set(self.id_downstream_data.keys())
        self.cif_ids = [cid for cid in self.cif_ids if cid in downstream_cif_keys]

    def load_dataset_split(self):
        # json_path = os.path.join(self.data_dir, 'dataset_split.json')
        
        json_path = os.path.join(self.data_dir, 'dataset_split.json')
        with open(json_path, 'r') as file:
            return json.load(file)      

    def get_prop(self, prop_file):
        assert os.path.exists(prop_file), f'{prop_file} does not exist!'
        print(f"reading {prop_file}")
        id_prop_data ={}
        with open(prop_file) as f:
            reader = csv.reader(f)
            for row in reader:
                key = row[0]
                try:
                    value = row[1]
                except ValueError:
                    continue
                id_prop_data[key] = value
        return id_prop_data
    
    def __len__(self) -> int:
        return len(self.cif_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = dict()
        cif_id = self.cif_ids[index]
        file_structure = os.path.join(self.pickle_path, f"{cif_id}.pickle")
        with open(file_structure, "rb") as f:
            crystal_data = pickle.load(f)
        data.update({"graph": crystal_data["graph"]})
        if self.compute_line_graph:
            data.update({"line_graph": crystal_data["line_graph"]})
        # get targets
        downstream_data=float(self.id_downstream_data[cif_id])
        data.update({"target": downstream_data,"cif_id":cif_id})
        return data

    @staticmethod
    def collate_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """batch collate function for JarvisDataset."""
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # get batch graph
        dict_batch["graph"] = dgl.batch(dict_batch["graph"])
        # get batch line graph
        if "line_graph" in keys:
            dict_batch["line_graph"] = dgl.batch(dict_batch["line_graph"])
        # get batch target
        dict_batch["target"] = torch.tensor(dict_batch["target"])
        return dict_batch
