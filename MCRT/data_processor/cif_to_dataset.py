import warnings
warnings.filterwarnings("ignore")
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from tqdm import tqdm
import os
import time
import json
import random
import numpy as np
import torch
import pickle
import argparse
import dgl
from typing import Optional, Tuple, Set
from collections import defaultdict
from jarvis.core.atoms import Atoms

'''
This file is used for constructing the dataset:1. symmetrize cifs to P1, 2. shuffle atoms 3. label atoms from which molecules, 4. compute bonds angle information, 5.generate Graphs with features to a pickle
'''

def main():
    parser = argparse.ArgumentParser(description='cif_path')
    parser.add_argument('--cif_path', type=str, help='input cif_path, the output will in this file too')
    parser.add_argument('--paral', type=int, help='if parallel, define num of max_workers.(int, >= 2)')
    parser.add_argument('--type', type=str, help='pretrain or finetune, default pretrain (more time-consuming)')
    args = parser.parse_args()
    if args.cif_path:
        cif_path = args.cif_path
    else:
        raise KeyError("No cif_path found")
    
    if  args.paral== None or args.paral >= 2 :
        paral = args.paral
    else:
        paral = False

    if  args.type:
        assert args.type in ['pretrain','finetune'],"--type should be pretrain or finetune"
        pickle_type = args.type
        print("generating for ",pickle_type)
    else:
        raise KeyError("--type should be provided")
    
    # Check if the source folder exists
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"Source folder not found at {cif_path}")
    
    cif_files = [os.path.join(cif_path, f) for f in os.listdir(cif_path) if f.endswith('.cif')]
    pickle_path = os.path.join(os.path.dirname(cif_path), "pickles")
    if os.path.exists(pickle_path):
        pickle_files = [os.path.join(cif_path, f) for f in os.listdir(pickle_path) if f.endswith('.pickle')]
        pickle_names = set(os.path.splitext(os.path.basename(f))[0] for f in pickle_files)
        cif_files = [f for f in cif_files if os.path.splitext(os.path.basename(f))[0] not in pickle_names]
        
    print('prepared cifs list')
    if not paral:
        for cif_file in tqdm(cif_files, desc="Identifying molecules"):
            parse_cif(cif_file,pickle_type)
    else:
        parse_cifs_in_parallel(cif_files, max_workers=paral,pickle_type=pickle_type)
    print('End parse cifs pickles' )

def jarvis_atoms_to_dgl_graph(
    atoms: Atoms,
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_canonize: bool = True,
) -> dgl.DGLGraph:
    """Convert JARVIS Atoms object to DGL graph.

    Args:
        atoms (Atoms): JARVIS Atoms object
        neighbor_strategy (str, optional): neighbor strategy. Defaults to "k-nearest".
        cutoff (float, optional): cutoff distance. Defaults to 8.0.
        max_neighbors (int, optional): maximum number of neighbors. Defaults to 12.
        use_canonize (bool, optional): whether to use canonize. Defaults to True.

    Returns:
        dgl.DGLGraph: DGL graph
    """
    # get node, edge, and edge distance
    if neighbor_strategy == "k-nearest":
        # get edges with k-nearest neighbors
        edges = nearest_neighbor_edges(
            atoms,
            cutoff,
            max_neighbors,
            use_canonize,
        )
        _u, _v, _r = build_undirected_edgedata(atoms, edges)
    elif neighbor_strategy == "radius_graph":
        pass  # not supported yet
    else:
        raise ValueError(
            f"neighbor_strategy must be one of k-nearest, radius_graph, but got {neighbor_strategy}"
        )
    # construct DGL graph
    graph = dgl.graph((_u, _v))
    # add node features
    graph.ndata["volume"] = torch.tensor([atoms.volume] * atoms.num_atoms)
    graph.ndata["coord"] = torch.tensor(atoms.cart_coords)
    graph.ndata["atomic_number"] = torch.tensor(atoms.atomic_numbers).long()
    # add edge features
    graph.edata["coord_diff"] = _r

    return graph
def nearest_neighbor_edges(
    atoms: Atoms,
    cutoff: float,
    max_neighbors: int,
    use_canonize: bool,
    max_attempts: int = 3,
):
    """Get edges with k-nearest neighbors.

    Args:
        atoms (Atoms): JARVIS Atoms object
        cutoff (float): cutoff distance
        max_neighbors (int): maximum number of neighbors
        use_canonize (bool): whether to use canonize
        max_attempts (int, optional): maximum number of attempts to find enough neighbors.
    Returns:
        edges (defaultdict[Tuple[int, int], Set[Tuple[float, float, float]]]): edges with images
    """
    # increase cutoff radius if minimum number of neighbors is higher than max_neighbors
    attempt = 0
    while True:
        # get all neighbors within the cutoff radius
        all_neighbors = atoms.get_all_neighbors(r=cutoff)

        # find the minimum number of neighbors
        min_nbrs = min(map(len, all_neighbors))

        # if there are fewer neighbors than the maximum allowed, increase the cutoff radius
        if min_nbrs < max_neighbors:
            # Calculate the new cutoff radius
            lat = atoms.lattice
            cutoff = max([lat.a, lat.b, lat.c, 2 * cutoff])
            attempt += 1
            if attempt > max_attempts:
                raise RuntimeError(
                    "Could not find enough neighbors to satisfy max_neighbors"
                )
        else:
            break

    # get edges with distance
    def canonize_edge(
        src_id,
        dst_id,
        src_image,
        dst_image,
    ):
        """Get canonical edge representation."""
        # store directed edges src_id <= dst_id
        if dst_id < src_id:
            src_id, dst_id = dst_id, src_id
            src_image, dst_image = dst_image, src_image

        # shift periodic images so that src is in (0,0,0) image
        if not np.array_equal(src_image, (0, 0, 0)):
            shift = src_image
            src_image = tuple(np.subtract(src_image, shift))
            dst_image = tuple(np.subtract(dst_image, shift))

        assert src_image == (0, 0, 0)

        return src_id, dst_id, src_image, dst_image

    edges = defaultdict(set)
    for site_idx, neighbors in enumerate(all_neighbors):
        # sort neighbors by distance
        neighbors = sorted(neighbors, key=lambda x: x[2])
        # get distance and neighbor indices and images
        distances = np.array([nbr[2] for nbr in neighbors])
        ids = np.array([nbr[1] for nbr in neighbors])
        images = np.array([nbr[3] for nbr in neighbors])

        # get the maximum distance with k-nearest neighbors
        max_dist = distances[max_neighbors - 1]

        # keep only neighbors within the cutoff radius
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        # get edges with images
        for dst, image in zip(ids, images):
            src_id, dst_id, _, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges
def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors.
    from jarvis.core.graphs import compute_bond_cosines
    """
    r1 = -edges.src["coord_diff"]
    r2 = edges.dst["coord_diff"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"angle": bond_cosine}  # a is edge features (bond angle cosines)
def build_undirected_edgedata(
    atoms: Atoms,
    edges: Optional[defaultdict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build undirected graph data from edge set.
    Implemented in ALIGNN.

    Args:
        atoms (Atoms): JARVIS Atoms object
        edges (Dict): edge set with images
    """
    _u, _v, _r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            dst = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])
            # add edges for both directions
            for _uu, _vv, _dd in [(src_id, dst_id, dst), (dst_id, src_id, -dst)]:
                _u.append(_uu)
                _v.append(_vv)
                _r.append(_dd)
    _u, _v, _r = (np.array(x) for x in (_u, _v, _r))
    _u = torch.tensor(_u)
    _v = torch.tensor(_v)
    _r = torch.tensor(_r).type(torch.get_default_dtype())

    return _u, _v, _r

def convert_structures_to_jarvis_atoms(structure: Structure) -> Atoms:
    return Atoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=[i.symbol for i in structure.species],
        cartesian=False,
    )
def read_structure(file_path):
    try:
        structure = Structure.from_file(file_path, occupancy_tolerance=100.0)
        # name = os.path.splitext(os.path.basename(file_path))[0]
        return structure, file_path
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def symmetry_operation(structures_with_names):
    structure, file_path = structures_with_names
    analyzer = SpacegroupAnalyzer(structure)
    symmetrized_structure = analyzer.get_symmetrized_structure()
    return symmetrized_structure,file_path

def shuffle_structure(structure,random_seed=123):
    # Create a list of indices and shuffle it
    indices = list(range(len(structure)))
    random.seed(random_seed)
    random.shuffle(indices)
    new_species = [structure[i].specie for i in indices]
    new_coords = [structure[i].frac_coords for i in indices]
    shuffled_structure = Structure(structure.lattice, new_species, new_coords)
    return shuffled_structure

def get_bonds_list(structure_graph):
    # Generate a list of bonds and their lengths
    bonds_list = []
    bond_id = 0
    for u, v in structure_graph.graph.edges():
        # Calculate the bond length and round it to 4 decimal places
        bond_length = round(structure_graph.structure.get_distance(u, v), 4)

        # Calculate the midpoint of the bond
        midpoint = (structure_graph.structure[u].coords + structure_graph.structure[v].coords) / 2

        # Add to the list (using atom indices as they appear in the CIF file)
        bonds_list.append((bond_id, (u, v), bond_length, midpoint))

        # Increment the bond ID
        bond_id += 1
    return bonds_list

def get_atom_graph(structure,radius=8,max_num_nbr=12):
    # RBF expand
    # gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr -
                                                     len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
    # nbr_fea = gdf.expand(nbr_fea)
    nbr_fea = torch.Tensor(nbr_fea) # it's actually distance now
    nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        
    return nbr_fea, nbr_fea_idx

def calculate_cosine_angle(v1, v2):
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_angle = np.clip(cosine_angle, -1, 1) #adjust wrong values due to limited precision
    return cosine_angle

# def get_nearest_neighbors_cos_angles(structure, n_nbr=8):
#     cos_angles_list = []
#     for i, site in enumerate(structure):
#             neighbors = structure.get_neighbors(site, r=max(structure.lattice.a,structure.lattice.b,structure.lattice.c)/2, include_index=True)
#             neighbors.sort(key=lambda x: x[1])
#             nearest_neighbors = neighbors[:n_nbr]
#             cos_angles = []
#             total_expected_angles = sum(range(n_nbr))
#             # Calculate actual angles
#             for j in range(n_nbr - 1):  # This ensures up to 7 loops for 8 neighbors
#                 actual_angles_this_round = 0
#                 for k in range(j + 1, len(nearest_neighbors)):
#                     vec1 = nearest_neighbors[j][0].coords - site.coords
#                     vec2 = nearest_neighbors[k][0].coords - site.coords
#                     cos_angle = calculate_cosine_angle(vec1, vec2)
#                     cos_angles.append(cos_angle)
#                     actual_angles_this_round += 1
                
#                 # Dynamic padding for missing angles in this round
#                 expected_angles_this_round = n_nbr - 1 - j
#                 missing_angles_this_round = expected_angles_this_round - actual_angles_this_round
#                 cos_angles.extend([-100] * missing_angles_this_round)

#             total_actual_angles = len(cos_angles)
#             if total_actual_angles < total_expected_angles:
#                 cos_angles.extend([-100] * (total_expected_angles - total_actual_angles))
#             cos_angles_list.append(cos_angles)

#     return cos_angles_list

# def get_nearest_neighbors_cos_angles(structure, n_nbr=8):
#     distance_matrix = structure.distance_matrix
#     cos_angles_list = []

#     for i in range(len(structure.sites)):
#         nearest_neighbors_indices = np.argsort(distance_matrix[i])[:n_nbr + 1]  # Include self, hence n_nbr+1
#         nearest_neighbors_indices = nearest_neighbors_indices[nearest_neighbors_indices != i]  # Exclude self

#         cos_angles = []
#         total_expected_angles = sum(range(n_nbr))

#         # Calculate actual angles
#         for j in range(n_nbr - 1):  # This ensures up to 7 loops for 8 neighbors
#             actual_angles_this_round = 0
#             for k in range(j + 1, n_nbr):
#                 vec1 = structure[nearest_neighbors_indices[j]].coords - structure[i].coords
#                 vec2 = structure[nearest_neighbors_indices[k]].coords - structure[i].coords
#                 cos_angle = calculate_cosine_angle(vec1, vec2)
#                 cos_angles.append(cos_angle)
#                 actual_angles_this_round += 1
            
#             # Dynamic padding for missing angles in this round
#             expected_angles_this_round = n_nbr - 1 - j
#             missing_angles_this_round = expected_angles_this_round - actual_angles_this_round
#             cos_angles.extend([-100] * missing_angles_this_round)

#         total_actual_angles = len(cos_angles)
#         if total_actual_angles < total_expected_angles:
#             cos_angles.extend([-100] * (total_expected_angles - total_actual_angles))
#         cos_angles_list.append(cos_angles)
#     # print(cos_angles_list)
#     return cos_angles_list

def get_nearest_neighbors_cos_angles(structure, n_nbr=8, distance_threshold=1e-3):

        def calculate_cosine_angle(v1, v2):
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_angle = np.clip(cosine_angle, -1, 1)  # Adjust for numerical stability
            return cosine_angle

        distance_matrix = structure.distance_matrix
        cos_angles_list = []

        for i in range(len(structure.sites)):
            # Step 1: Sort all neighbors by distance
            all_indices_sorted = np.argsort(distance_matrix[i])
            all_distances_sorted = distance_matrix[i, all_indices_sorted]

            # Exclude the current point itself
            all_indices_sorted = all_indices_sorted[all_indices_sorted != i]
            all_distances_sorted = all_distances_sorted[1:]  # Skip the first element (self)

            # Step 2: Determine the distance of the nth nearest neighbor
            n_nbr_distance = all_distances_sorted[n_nbr - 1]  # Distance of the nth neighbor

            # Step 3: Extend neighbors if further points have a distance within the threshold
            extended_count = n_nbr
            while extended_count < len(all_distances_sorted) and abs(all_distances_sorted[extended_count] - n_nbr_distance) < distance_threshold:
                extended_count += 1

            # Step 4: Select the extended list of nearest neighbors
            nearest_neighbors_indices = all_indices_sorted[:extended_count]
            nearest_distances = all_distances_sorted[:extended_count]

            # Step 5: Group neighbors by identical distances (within threshold)
            distance_groups = []
            group_indices = []
            
            for j in range(len(nearest_distances)):
                if not any(abs(nearest_distances[j] - group[0]) < distance_threshold for group in distance_groups):
                    indices = np.where(abs(nearest_distances - nearest_distances[j]) < distance_threshold)[0]
                    distance_groups.append(nearest_distances[indices])
                    group_indices.append(indices)
            
            # Step 6: Precompute the cumulative middle vector and internal average cosine for each group
            group_middle_vectors = {}
            group_internal_cos_angles = {}

            for idx, group in enumerate(group_indices):
                if len(group) > 1:
                    vectors = np.array([structure[nearest_neighbors_indices[g]].coords - structure[i].coords for g in group])
                    
                    # Normalize and accumulate vectors to get an average direction
                    unit_vectors = np.array([vec / np.linalg.norm(vec) for vec in vectors])
                    middle_vector = np.sum(unit_vectors, axis=0)
                    
                    # Calculate internal squared average angle
                    angles = []
                    for m in range(len(vectors)):
                        for n in range(m + 1, len(vectors)):
                            angle = np.arccos(calculate_cosine_angle(vectors[m], vectors[n]))
                            angles.append(angle ** 2)
                    avg_squared_angle = np.sqrt(np.mean(angles))
                    group_internal_cos_angles[idx] = np.cos(avg_squared_angle)  # Cosine of internal average angle
                    group_middle_vectors[idx] = middle_vector if np.linalg.norm(middle_vector) > 1e-6 else np.array([0.0, 0.0, 0.0])
                else:
                    group_middle_vectors[idx] = structure[nearest_neighbors_indices[group[0]]].coords - structure[i].coords
                    group_internal_cos_angles[idx] = 1  # Single atom, angle is zero

            # Step 7: Calculate cosines using the modified logic with cumulative middle vectors and internal angles
            cos_angles = []
            total_expected_angles = sum(range(n_nbr))  # Total expected angles for n_nbr neighbors

            for j in range(n_nbr - 1):  # Ensure n_nbr - 1 loops for dynamic padding
                actual_angles_this_round = 0
                for k in range(j + 1, n_nbr):
                    if j < len(nearest_neighbors_indices) and k < len(nearest_neighbors_indices):
                        group_j = next((idx for idx, g in enumerate(group_indices) if j in g), None)
                        group_k = next((idx for idx, g in enumerate(group_indices) if k in g), None)
                        
                        if group_j is not None and group_k is not None:
                            if group_j == group_k:
                                cos_angle = group_internal_cos_angles[group_j]
                            else:
                                cos_angle = -100 if np.linalg.norm(group_middle_vectors[group_j]) < 1e-6 or np.linalg.norm(group_middle_vectors[group_k]) < 1e-6 else calculate_cosine_angle(group_middle_vectors[group_j], group_middle_vectors[group_k])
                        elif group_j is not None:
                            vec_k = structure[nearest_neighbors_indices[k]].coords - structure[i].coords
                            cos_angle = -100 if np.linalg.norm(group_middle_vectors[group_j]) < 1e-6 else calculate_cosine_angle(group_middle_vectors[group_j], vec_k)
                        elif group_k is not None:
                            vec_j = structure[nearest_neighbors_indices[j]].coords - structure[i].coords
                            cos_angle = -100 if np.linalg.norm(group_middle_vectors[group_k]) < 1e-6 else calculate_cosine_angle(vec_j, group_middle_vectors[group_k])
                        else:
                            vec1 = structure[nearest_neighbors_indices[j]].coords - structure[i].coords
                            vec2 = structure[nearest_neighbors_indices[k]].coords - structure[i].coords
                            cos_angle = calculate_cosine_angle(vec1, vec2)
                        
                        cos_angles.append(cos_angle)
                        actual_angles_this_round += 1
                    else:
                        cos_angles.append(-100)  # Padding for missing neighbors

                expected_angles_this_round = n_nbr - 1 - j
                missing_angles_this_round = expected_angles_this_round - actual_angles_this_round
                cos_angles.extend([-100] * missing_angles_this_round)

            total_actual_angles = len(cos_angles)
            if total_actual_angles < total_expected_angles:
                cos_angles.extend([-100] * (total_expected_angles - total_actual_angles))
            
            cos_angles_list.append(cos_angles)

        return cos_angles_list
    
def identify_and_label_molecules(symmetrized_structure_with_name,pickle_type='pretrain'):
    """
    Identify and label molecules in a crystal structure using StructureGraph.
    Each disconnected subgraph in the undirected graph is considered a separate molecule.
    dimensions:
    atom_num:[n,1]; nbr_fea:[n,num_nbr,41]; nbr_fea_idx:[n,12]; distance_matrix:[n,n]; atom_to_molecule_list: [n,1]; 
    angles_list:[n,n_nbr*(n_nbr-1)/2].
    """
    structure, file_path = symmetrized_structure_with_name
    cif_id = os.path.splitext(os.path.basename(file_path))[0]
    if structure.num_sites > 1023:
        print("cif",cif_id,": Atom length {structure.num_sites} > 1023, please remove it, will not build its graph to pickle")
    else:
        
        if pickle_type=='pretrain':
            # build graph
            structure_graph = StructureGraph.with_local_env_strategy(structure, JmolNN())
            # bonds_list = get_bonds_list(structure_graph)#[(bond_id, (u, v), bond_length, midpoint)]
            undirected_graph = structure_graph.graph.to_undirected()
            molecules = [undirected_graph.subgraph(c).copy() for c in nx.connected_components(undirected_graph)]
            atom_to_molecule = {}
            for i, mol in enumerate(molecules, start=1):
                for node in mol.nodes:
                    atom_to_molecule[int(node)] = i
            atm_list = [atom_to_molecule[k] for k in sorted(atom_to_molecule, key=int)]
            # Extract the species (atom types) from the structure
            atom_num = [site.specie.Z for site in structure] #store the atom number starting from 1!
            angles_list=get_nearest_neighbors_cos_angles(structure=structure, n_nbr=8)# very time consuming

            crystal_data={
                    "structure":structure,
                    "atom_num":atom_num,
                    "atm_list":atm_list,
                    "angles_list":angles_list,
                    "cif_id":cif_id,
                }
        elif pickle_type=='finetune':
            # Extract the species (atom types) from the structure
            atom_num = [site.specie.Z for site in structure] #store the atom number starting from 1!
            angles_list=get_nearest_neighbors_cos_angles(structure=structure, n_nbr=8)
            # atoms = convert_structures_to_jarvis_atoms(structure)
            # graph = jarvis_atoms_to_dgl_graph(
            #             atoms,
            #             "k-nearest",
            #             8.0,
            #             12,
            #             True,
            #         )
            # graph.apply_edges(
            #             lambda edges: {"distance": torch.norm(edges.data["coord_diff"], dim=1)}
            #         )
            # line_graph = graph.line_graph(shared=True)
            # line_graph.apply_edges(compute_bond_cosines)
            crystal_data={
                    "structure":structure,
                    "atom_num":atom_num,
                    "angles_list":angles_list,
                    # "graph":graph,
                    # "line_graph":line_graph,
                    "cif_id":cif_id,
                }
        else:
            print("--type should be pretrain or finetune")
        # save pickles    
        pickle_path=os.path.join(os.path.dirname(os.path.dirname(file_path)),"pickles")
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        with open(os.path.join(pickle_path, f"{cif_id}.pickle"), "wb") as file:
            pickle.dump(crystal_data, file) 

      

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self,dmin=0, dmax=8, step=0.2, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

def parse_cif(file_path, pickle_type='pretrain'):
    try:
        structures_with_names=read_structure(file_path)
        symmetrized_structure_with_name=symmetry_operation(structures_with_names)
        identify_and_label_molecules(symmetrized_structure_with_name, pickle_type)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        pass
    
def parse_cifs_in_parallel(file_paths, max_workers, pickle_type='pretrain'):
    """
    Process multiple structures in parallel using ProcessPoolExecutor.
    """
    start_time = time.time()
    pickle_types = [pickle_type for _ in range(len(file_paths))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(file_paths), desc="Identifying molecules in parallel") as pbar:
            futures = {executor.submit(parse_cif, file_path, pickle_type): file_path for file_path, pickle_type in zip(file_paths, pickle_types)}
            dataset = []
            for future in as_completed(futures):
                result = future.result()
                dataset.append(result)
                pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    structures_per_second = len(file_paths) / elapsed_time

    print(f"identify_molecules_Processed {len(file_paths)} structures in {elapsed_time:.2f} seconds.")
    print(f"identify_molecules_Average speed: {structures_per_second:.2f} structures/second")
    return dataset

if __name__ == "__main__":
    main()
