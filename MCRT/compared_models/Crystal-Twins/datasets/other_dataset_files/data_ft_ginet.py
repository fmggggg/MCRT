import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import ase
# from ase.io import cif
from ase.io import read as ase_read
from pymatgen.core.structure import Structure
from sklearn import preprocessing
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_geometric.data import Data, Dataset, DataLoader
from torch_cluster import knn_graph

from datasets.atom_feat import AtomCustomJSONInitializer


# ATOM_LIST = list(range(1,119))
# CHIRALITY_LIST = [
#     Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#     Chem.rdchem.ChiralType.CHI_OTHER
# ]
# ATOM_LIST = [
#     1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
#     31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 
#     58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
#     83, 89, 90, 91, 92, 93, 94
# ]
ATOM_LIST = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
    83, 89, 90, 91, 92, 93, 94
]
print("Number of atoms:", len(ATOM_LIST))


# def get_all_cifs(data_dir):
#     cif_ids = []
#     for subdir, dirs, files in os.walk(data_dir):
#         for fn in files:
#             if fn.endswith('.cif'):
#                 cif_ids.append(fn)
#     return cif_ids


def read_csv(csv_dir, task):
    csv_path = os.path.join(csv_dir, 'id_prop_m.csv')
    cif_ids, labels = [], []
    MP_energy = False
    if 'MP-formation-energy' in csv_dir:
        MP_energy = True

    if 'perovskites' in csv_dir:
        Perovskites = True

    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                cif_id = row['CIF_ID']
                label = row['val']
                if MP_energy:
                    cif_ids.append(cif_id+'.cif')
                if Perovskites:
                    cif_ids.append(cif_id+'.cif')
                else:
                    cif_ids.append(cif_id)

                if task == 'classification':
                    label = int(label == 'True')
                    labels.append(label)
                    # labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    ValueError('task must be either regression or classification')
    return cif_ids, labels


class CrystalDataset(Dataset):
    def __init__(self, data_dir='data/BG_cifs', k=5, task='regression'):
        super(Dataset, self).__init__()
        self.k = k
        self.task = task
        self.data_dir = data_dir
        self.cif_ids, self.labels = read_csv(data_dir, task)
        self.labels = np.array(self.labels)
        # cif_ids_csv, labels_csv = read_csv(data_dir, task)
        # # print(cif_ids_csv)
        # cif_ids_file = set(get_all_cifs(data_dir))
        # # print(cif_ids_file)
        # self.cif_ids, self.labels = [], []
        # for cif_id, label in zip(cif_ids_csv, labels_csv):
        #     if cif_id in cif_ids_file:
        #         self.cif_ids.append(cif_id)
        #         self.labels.append(label)
        print(len(self.labels))
        self.atom_featurizer = AtomCustomJSONInitializer(os.path.join(self.data_dir,'atom_init.json'))
        self.feat_dim = self.atom_featurizer.get_length()

    def __getitem__(self, index):
        # get the cif id and path
        cif_id = self.cif_ids[index]
        cryst_fn = os.path.join(self.data_dir, cif_id)
        
        if self.task == 'regression':
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(self.labels.reshape(-1,1))
            self.labels = self.scaler.transform(self.labels.reshape(-1,1))
        # # read cif using ASE
        # crys = ase_read(cryst_path)
        # atom_indices = crys.numbers
        # pos = crys.positions
        # feat = self.atom_featurizer.get_atom_features(atom_indices)
        # N = len(pos)

        # read cif using pymatgen
        crys = Structure.from_file(cryst_fn)        
        pos = crys.frac_coords
        atom_indices = list(crys.atomic_numbers)
        cell = crys.lattice.get_cartesian_coords(1)
        feat = self.atom_featurizer.get_atom_features(atom_indices)
        N = len(pos)

        y = self.labels[index]
        if self.task == 'regression':
            y = torch.tensor(y, dtype=torch.float).view(1,1)
        elif self.task == 'classification':
            # y = torch.tensor(y, dtype=torch.long).view(1,1)
            y = torch.tensor(y, dtype=torch.float).view(1,1)

        # build the PyG graph 
        atomics = []
        for idx in atom_indices:
            atomics.append(ATOM_LIST.index(idx))
        atomics = torch.tensor(atomics, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        feat = torch.tensor(feat, dtype=torch.float)
        edge_index = knn_graph(pos, k=self.k, loop=False)
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
        
        # return PyG Data object
        data = Data(
            atomics=atomics, pos=pos, feat=feat, y=y, 
            edge_index=edge_index, edge_attr=edge_attr
        )
        return data

    def __len__(self):
        return len(self.cif_ids)


class CrystalDatasetWrapper(object):
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, 
        data_dir='data', k=3, task='regression'
    ):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.task = task

    def get_data_loaders(self):
        train_dataset = CrystalDataset(self.data_dir, self.k, self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        print(num_train)

        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        split2 = int(np.floor(self.test_size * num_train))
        # train_idx, valid_idx, test_idx = indices[split:], indices[:split], indices
        valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)
                                
        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                  num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    dataset = CrystalDataset()
    print(dataset)
    print(dataset.__getitem__(0))
    dataset = CrystalDatasetWrapper(batch_size=2, num_workers=0, valid_size=0.1, test_size=0.1, data_dir='data/BG_cifs')
    train_loader, valid_loader, test_loader = dataset.get_data_loaders()
    for bn, data in enumerate(train_loader):
        print(data)
        print(data.atomics)
        print(data.pos)
        print(data.y)
        break