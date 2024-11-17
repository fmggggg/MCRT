import copy
import math
import ase.io
from ase.build import make_supercell
from itertools import product
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable
import numpy as np
import os
import pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
import torch
import pytorch_lightning as pl
from MCRT.modules.McrtModule import MCRT_Module
from MCRT.modules.datamodule import Datamodule
from MCRT.config import config
import csv

@lru_cache
def get_model_and_datamodule(model_path, data_root, downstream=""):
    _config = config()
    _config["visualize"] = True
    _config["per_gpu_batchsize"] = 1
    _config["data_root"] = data_root
    _config["root_dataset"] = data_root
    _config["load_path"] = model_path
    _config["test_only"] = True
    _config["log_dir"] = "result_visualization"
    _config["downstream"] = downstream
    _config["num_workers"] = 0

    pl.seed_everything(_config["seed"])
    model = MCRT_Module(_config)
    model.setup("test")
    model.eval()
    model.to("cpu")

    dm = Datamodule(_config)
    dm.setup("test")
    data_iter = dm.test_dataloader()

    return model, data_iter


@lru_cache
def get_batch_from_index(data_iter, batch_id):
    iter_ = iter(data_iter)
    for _ in range(batch_id):
        next(iter_)
    batch = next(iter_)
    return batch


@lru_cache
def get_batch_from_cif_id(data_iter, cif_id):
    cif_id = Path(cif_id).stem
    iter_ = iter(data_iter)
    while True:
        try:
            batch = next(iter_)
        except StopIteration:
            raise ValueError(f"There are no {cif_id} in dataset")
        else:
            batch_id = batch["cif_id"][0]
            if batch_id == cif_id:
                return batch


@lru_cache
def get_primitive_structure(path_cif, tolerance=2.0):
    (st,) = CifParser(path_cif, occupancy_tolerance=tolerance).get_structures(
        primitive=True
    )
    return st

@lru_cache
def get_structure(
    path_cif, make_supercell=False, dtype="ase", *, max_length=60, min_length=15
):
    """
    get primitive structure from path_cif
    :param path_cif: <str> path for cif file
    :param make_supercell: <bool> if True,
    :param dtype: <str> -> ['ase', 'pymatgen'] return type for structure.
    :param max_length: <int/float> max p_lattice length of structure file (Å)
    :param min_length: <int/float> min p_lattice length of structure file (Å)
    :return: <pymatgen.Structure> structure file from path cif
    """
    def transform_path_cif_to_pickle(path_cif):
        directory, filename = os.path.split(path_cif)
        new_filename = os.path.splitext(filename)[0] + ".pickle"
        parent_directory = os.path.dirname(directory)
        new_directory = os.path.join(parent_directory, "pickles")
        new_path = os.path.join(new_directory, new_filename)
        return new_path
    file_structure = transform_path_cif_to_pickle(path_cif)

    with open(file_structure, "rb") as f:
        crystal_data = pickle.load(f)
    structure=crystal_data["structure"]

    # structure.to(fmt="cif", filename="MCRT/visualize/results/output.cif")

    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)

    if make_supercell:
        atoms = get_supercell_structure(atoms, max_length, min_length)
    # else:
    #     atoms = get_supercell_structure(atoms, max_length, 4)

    if dtype == "pymatgen":
        return AseAtomsAdaptor().get_structure(atoms)
    elif dtype == "ase":
        return atoms
    else:
        raise TypeError(f"type must be ase or pymatgen, not {dtype}")


def get_supercell_structure(atoms, max_length=60, min_length=15):
    """
    get supercell atoms from <ase.Atoms>
    :param atoms: <ase.Atoms> object
    :param max_length: <int/float> max p_lattice length of structure file (Å)
    :param min_length: <int/float> min p_lattice length of structure file (Å)
    :return: <ase.Atoms or pymatgen.Structure> structure type.
    """
    scale_abc = []
    for l in atoms.cell.cellpar()[:3]:
        if l > max_length:
            raise ValueError(
                f"primitive p_lattice is larger than max_length {max_length}"
            )
        elif l < min_length:
            scale_abc.append(math.ceil(min_length / l))
        else:
            scale_abc.append(1)

    m = np.zeros([3, 3])
    np.fill_diagonal(m, scale_abc)
    atoms = make_supercell(atoms, m)
    return atoms


def cuboid_data(position, color=None, num_patches=(6, 6, 6), lattice=None):
    """
    Get cuboid plain data from position and size data
    :param position: <list/tuple> patch positions => [x, y, z]
    :param color: <list/tuple> colors => [r, g, b, w]
    :param num_patches: number of patches in each axis (default : (6, 6, 6))
    :param lattice: <np.ndarray> p_lattice vector for unit p_lattice
    :return: <tuple> (list of plain vector, list of color vector)
    """
    if isinstance(num_patches, (tuple, list)):
        num_patches = np.array(num_patches)
    elif not isinstance(num_patches, np.ndarray):
        raise TypeError(f"num_patches must be tuple or list, not {type(num_patches)}")

    bound = np.array([[0, 1] for _ in range(3)]) + np.array(position)[:, np.newaxis]
    vertex = np.array(list(product(*bound)))
    plane_ls = []
    for i, (dn, up) in enumerate(bound):
        plane1 = np.matmul(vertex[vertex[:, i] == dn], lattice / num_patches)
        plane2 = np.matmul(vertex[vertex[:, i] == up], lattice / num_patches)

        plane_ls.append(plane1)
        plane_ls.append(plane2)

    plane_ls = np.array(plane_ls).astype("float")
    plane_ls[:, [0, 1], :] = plane_ls[:, [1, 0], :]

    color_ls = np.repeat(color[np.newaxis, :], 6, axis=0)

    return plane_ls, color_ls


def plot_cube(positions, colors, lattice, num_patches=(6, 6, 6), **kwargs):
    """
    help function for draw 3d cube plot
    :param positions: <list> list of patch position
    :param colors: <list -> list> list of color codes [r, g, b, w]
    :param lattice: <np.ndarray> p_lattice vector for unit p_lattice
    :param num_patches: number of patches in each axis (default : (6, 6, 6))
    :param kwargs: kwargs for <matplotlib.Poly3DCollection>
    :return: <matplotlib.Poly3DCollection> cuboid matplotlib object
    """

    data = [
        cuboid_data(pos, color, num_patches=num_patches, lattice=lattice)
        for pos, color in zip(positions, colors)
    ]
    plain_ls, color_ls = zip(*data)

    return Poly3DCollection(
        np.concatenate(plain_ls), facecolors=np.concatenate(color_ls), **kwargs
    )


def get_heatmap(out, batch_idx, graph_len=1023,image_len=100, skip_cls=True):
    """
    attention rollout  in "Quantifying Attention Flow in Transformers" paper.
    :param out: output of model.infer(batch)
    :param batch_idx: batch index
    :param graph_len: the length of grid embedding
    :param skip_cls: <bool> If True, class token is ignored.
    :return: <np.ndarray> heatmap graph, heatmap grid
    """
    print("start cal heatmap")
    attn_weights = torch.stack(
        out["attn_weights"]
    )  # [num_layers, B, num_heads, max_len, max_len]
    att_mat = attn_weights[:, batch_idx]  # [num_layers, num_heads, max_len, max_len]

    image=out["image"]# list 2 [B,1,h,w]
    max_birth_1d=out["max_birth_1d"]#list [B]
    max_persistence_1d=out["max_persistence_1d"]
    max_birth_2d=out["max_birth_2d"]
    max_persistence_2d=out["max_persistence_2d"]
    # transform into numpy
    image_1d=image[0][batch_idx, 0, :, :].detach().numpy() #[h,w]
    image_2d=image[1][batch_idx, 0, :, :].detach().numpy() #[h,w]
    max_birth_1d=max_birth_1d[batch_idx].detach().numpy()
    max_persistence_1d=max_persistence_1d[batch_idx].detach().numpy()
    max_birth_2d=max_birth_2d[batch_idx].detach().numpy()
    max_persistence_2d=max_persistence_2d[batch_idx].detach().numpy()

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)  # [num_layers, max_len, max_len]

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att

    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(
        -1
    )  # [num_layers, max_len, max_len]
    aug_att_mat = aug_att_mat.detach().numpy()  # prevent from memory leakage

    # Recursively multiply the weight matrices
    joint_attentions = np.zeros(aug_att_mat.shape)  # [num_layers, max_len, max_len]
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.shape[0]):
        joint_attentions[n] = np.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]  # [max_len, max_len]

    # Don't drop class token when normalizing
    if skip_cls:
        v_ = v[0][1:]  # skip cls token
        cost_graph = v_[:graph_len]  # / v_.max()
        cost_image = v_[graph_len:]  # / v_.max()
        heatmap_graph = cost_graph
        heatmap_image_1d = cost_image[1:image_len+3]  # omit sep tokens
        heatmap_image_2d = cost_image[image_len+3:]  # omit scale tokens
    else:
        v_ = v[0]
        cost_graph = v_[: graph_len + 1]  # / v_.max()
        cost_image = v_[graph_len + 1 :]  # / v_.max()
        heatmap_graph = cost_graph[1:]  # omit cls token
        heatmap_image_1d = cost_image[1:image_len+3] # omit sep tokens
        heatmap_image_2d = cost_image[image_len+3:] # omit scale tokens

    return heatmap_graph, heatmap_image_1d, heatmap_image_2d,image_1d,image_2d,max_birth_1d,max_persistence_1d,max_birth_2d,max_persistence_2d,


def scaler(value, min_att, max_att):
    if isinstance(value, float):
        if value > max_att:
            value = max_att
        elif value < min_att:
            value = min_att
        return float((value - min_att) / (max_att - min_att))

    elif isinstance(value, np.ndarray):
        value = copy.deepcopy(value)
        value[value > max_att] = max_att
        value[value < min_att] = min_att
        return (value - min_att) / (max_att - min_att)
    elif isinstance(value, Iterable):
        return scaler(np.array(list(value), dtype="float"), min_att, max_att)
    else:
        raise TypeError(f"value must be float, list, or np.array, not {type(value)}")
