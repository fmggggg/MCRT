import warnings
warnings.filterwarnings("ignore")
import sys
import os
import json
import random
import numpy as np
import torch
import pickle
import csv
import dgl
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from MCRT.data_processor.cif_to_dataset import (
    jarvis_atoms_to_dgl_graph,
    compute_bond_cosines,
    convert_structures_to_jarvis_atoms,
)
'''
This file is used for constructing the dataset for pytorch
'''  

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        max_num_nbr: int=12,
        nbr_fea_len: int=41,
        num_ap:int=200,
        n_dist:int=100,
        n_angle:int=100,
        downstream="",
        tasks=[],
        if_conv=False,
        pos_emb='relative',
        if_alignn = True,
        if_image = True,
        if_grid = False,
        read_from_pickle = True,
    ):
        """
        Dataset for pretrained MCRT.
        Args:
            data_dir (str): where dataset cif files
        """
        super().__init__()
        self.data_dir = data_dir
        assert split in {"train", "test", "val"}
        self.split = split
        self.max_num_nbr = max_num_nbr
        self.nbr_fea_len = nbr_fea_len
        self.num_ap=num_ap
        self.n_dist=n_dist
        self.n_angle=n_angle
        self.downstream=downstream
        self.tasks=tasks
        self.if_conv=if_conv
        self.pos_emb=pos_emb
        self.if_alignn=if_alignn
        self.if_image = if_image
        self.if_grid = if_grid
        self.read_from_pickle = read_from_pickle

        if "cdp" in self.tasks:
            density_file = os.path.join(self.data_dir, 'density.csv')
            self.id_density_data=self.get_prop(density_file)
        if "sgp" in self.tasks:
            space_group_file = os.path.join(self.data_dir, 'space_group.csv')
            self.id_space_group_data=self.get_prop(space_group_file)
        if "sep" in self.tasks: 
            space_group_file = os.path.join(self.data_dir, 'space_group.csv')
            self.id_space_group_data=self.get_prop(space_group_file)
            symm_elem_file = os.path.join(self.data_dir, 'simplyfied_symm_elem.json')
            with open(symm_elem_file, 'r') as f:
                data_dict = json.load(f)
            self.symm_elem_data=data_dict

        self.cif_path=os.path.join(self.data_dir,"cifs")

        if self.read_from_pickle:
            self.pickle_path=os.path.join(self.data_dir,"pickles") # pickles include the info for atom_label (for "map") and atm_label (for "apc")
            if not os.path.exists(self.pickle_path):
                print('!!!pickle path does not exist, generate graph input for finetune in real time. NOTE:it doesnot support pretrain because label for pretrain is time-consuming!!!')
                self.read_from_pickle=False
                assert os.path.exists(self.cif_path), 'cifs does not exist!'

        if if_image:
            self.image_path=os.path.join(self.data_dir,"imgs") 
            assert os.path.exists(self.image_path), 'images does not exist!'

        if if_grid:
            self.grid_path=os.path.join(self.data_dir,"grids") 
            assert os.path.exists(self.grid_path), 'grids does not exist!'

        dataset_split = self.load_dataset_split()


        conditions = [] # restore all cif conditions
        if self.read_from_pickle:
            pickle_cif_ids = set(os.path.splitext(name)[0] for name in os.listdir(self.pickle_path) if name.endswith('.pickle'))
            conditions.append(pickle_cif_ids)

        if if_image:
            image_cif_ids = set(os.path.splitext(name)[0] for name in os.listdir(self.image_path) if name.endswith('.npy'))
            conditions.append(image_cif_ids)

        if if_grid:
            grid_cif_ids = set(os.path.splitext(name)[0] for name in os.listdir(self.grid_path) if name.endswith('.grid'))
            griddata16_cif_ids = set(os.path.splitext(name)[0] for name in os.listdir(self.grid_path) if name.endswith('.griddata16'))
            combined_grid_ids = grid_cif_ids & griddata16_cif_ids
            conditions.append(combined_grid_ids)

        # Combine all conditions
        if conditions:
            combined_conditions = set.intersection(*conditions)
            self.cif_ids = [cid for cid in dataset_split[self.split] if cid in combined_conditions]
        else:
            self.cif_ids = dataset_split[self.split]

        if self.downstream:
            downstream_file = os.path.join(self.data_dir, f"{downstream}.csv")
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

    def __len__(self):
        return len(self.cif_ids)

    @staticmethod
    def apply_mask_to_atom_num(atom_num, mask_probability=0.15):
        """
        Applies masking logic directly to atom numbers in the dataset.

        Args:
        atom_num (torch.Tensor): Tensor of atom types/numbers, labels for each atom.
        mask_probability (float): Probability of applying mask.

        Returns:
        masked_atom_num (torch.Tensor): Tensor of atom numbers after applying mask.
        atom_label (torch.Tensor): Labels tensor for loss computation, with -1 indicating atoms to ignore.
        """
        atom_label = torch.full_like(atom_num, -1)
        mask = torch.rand(atom_num.size(0)) < mask_probability
        masked_atom_num = atom_num.clone()

        for i in range(masked_atom_num.size(0)):
            if mask[i]:
                atom_label[i] = atom_num[i]# Set the correct label for masked atom
                decision = torch.rand(1).item()
                if decision < 0.8:  # 80% chance to replace with [MASK], in this case 0
                    masked_atom_num[i] = 0
                elif decision < 0.9:  # 10% chance for random replacement
                    masked_atom_num[i] = torch.randint(1, 119, (1,)).item()
                # Remaining 10% chance to keep unchanged
                
        return masked_atom_num, atom_label  

    def select_atom_pairs(self, atm_list, m):
        """    
        m: number of atom pairs from each crystal
        half of m atom pairs are from same molecule, remaining from different molecules,
        if possible number of atom pairs < m, number of selected pairs = max_pairs*2, max_pairs = min(m // 2, same_molecule_pairs_count, diff_molecule_pairs_count)
        """
        atom_pairs = torch.full((m, 2), -1, dtype=torch.long)  # Use -1 for invalid pairs
        ap_labels = torch.full((m,), -1, dtype=torch.long)  # Use -1 for invalid ap_labels

        
        # Collect information about molecules in the crystal
        molecule_atoms = {}
        for idx, molecule_id in enumerate(atm_list):
            molecule_atoms.setdefault(molecule_id, []).append(idx)

        # Calculate the number of possible same molecule pairs
        same_molecule_pairs_count = sum(len(atoms) * (len(atoms) - 1) // 2 for atoms in molecule_atoms.values())
        # Calculate the number of possible diff molecule pairs
        diff_molecule_pairs_count = 0
        molecule_ids = list(molecule_atoms.keys())
        for idx1, mol1 in enumerate(molecule_ids):
                for mol2 in molecule_ids[idx1 + 1:]:
                    diff_molecule_pairs_count += len(molecule_atoms[mol1]) * len(molecule_atoms[mol2])

        max_pairs = min(m // 2, same_molecule_pairs_count, diff_molecule_pairs_count)

        selected_pairs = []
        selected_pairs_set = set()

            # Randomly select same molecule pairs
        while len(selected_pairs) < max_pairs:
                mol = random.choice(list(molecule_atoms.keys()))
                if len(molecule_atoms[mol]) > 1:
                    atom1, atom2 = random.sample(molecule_atoms[mol], 2)
                    pair = tuple(sorted((atom1, atom2)))  # Ensure the pair is sorted
                    if pair not in selected_pairs_set:
                        selected_pairs.append(pair)
                        ap_labels[len(selected_pairs) - 1] = 1  # Same molecule
                        selected_pairs_set.add(pair)

            # Select different molecule pairs
        while len(selected_pairs) < 2 * max_pairs:
                mol1, mol2 = random.sample(list(molecule_atoms.keys()), 2)
                atom1, atom2 = random.choice(molecule_atoms[mol1]), random.choice(molecule_atoms[mol2])
                pair = tuple(sorted((atom1, atom2)))  # Ensure the pair is sorted
                if pair not in selected_pairs_set:
                    selected_pairs.append(pair)
                    ap_labels[len(selected_pairs) - 1] = 0  # Different molecules
                    selected_pairs_set.add(pair)

            # atom_pairs[:len(selected_pairs)] = torch.tensor(selected_pairs)
        if len(selected_pairs) > 0:  # Ensure there are selected pairs
                    selected_pairs_tensor = torch.tensor(selected_pairs, dtype=torch.long)
                    atom_pairs[:selected_pairs_tensor.size(0)] = selected_pairs_tensor

        return atom_pairs, ap_labels #  [m, 2],[m] 
    
    def select_dist_pairs(self, distance_matrix, n_dist=100):
        dist_pairs = torch.full((n_dist, 2), -1, dtype=torch.long)
        dist_labels = torch.full((n_dist,), -1.0, dtype=torch.float)
        n_atoms = distance_matrix.shape[0]
        # possible pairs
        max_possible_pairs = n_atoms * (n_atoms - 1) // 2
        max_n_dist = min(n_dist, max_possible_pairs)


        dist_pairs_set = set()
        pair_count = 0

        while pair_count < max_n_dist:
            i = random.randint(0, n_atoms - 1)
            j = random.randint(0, n_atoms - 1)
            if i != j:
                pair = tuple(sorted((i, j)))
                if pair not in dist_pairs_set:
                    dist_pairs[pair_count] = torch.tensor(pair, dtype=torch.long)
                    dist_labels[pair_count] = distance_matrix[i, j]
                    dist_pairs_set.add(pair)
                    pair_count += 1
        return dist_pairs, dist_labels #  [n_dist, 2],[n_dist] 
    
    def select_angle_pairs(self, structure, n_angle=100):
        angle_pairs = torch.full((n_angle, 3), -1, dtype=torch.long)
        angle_labels = torch.full((n_angle,), -100.0, dtype=torch.float)
        n_atoms = len(structure)
        # possible triples
        max_possible_triples = n_atoms * (n_atoms - 1) * (n_atoms - 2) // 6
        max_n_angle = min(n_angle, max_possible_triples)

        angle_pairs_set = set()
        pair_count = 0

        while pair_count < max_n_angle:
            atoms = random.sample(range(n_atoms), 3)
            triple = tuple(atoms)
            if triple not in angle_pairs_set:
                # Calculate the angle
                vec1 = structure[atoms[1]].coords - structure[atoms[0]].coords
                vec2 = structure[atoms[1]].coords - structure[atoms[2]].coords
                cos_angle = self.calculate_cosine_angle(vec1, vec2)

                angle_pairs[pair_count] = torch.tensor(triple, dtype=torch.long)
                angle_labels[pair_count] = cos_angle
                angle_pairs_set.add(triple)
                pair_count += 1

        return angle_pairs, angle_labels

    @staticmethod
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
    
    @staticmethod
    def get_gaussian_distance(distances, num_step:int=41, dmax=8, dmin=0, var=None):
        """
        Expands the distance by Gaussian basis, adapted from (https://github.com/txie-93/cgcnn.git)
        """

        assert dmin < dmax
        _filter = np.linspace(
            dmin, dmax, num_step
        ) 
        if var is None:
            var = (dmax-dmin)/(num_step-1)
        return np.exp(-((distances[..., np.newaxis] - _filter) ** 2) / var**2)
    
    
    @staticmethod
    def shuffle_structure(structure):
        # Create a list of indices and shuffle it
        indices = list(range(len(structure)))
        random.shuffle(indices)
        new_species = [structure[i].specie for i in indices]
        new_coords = [structure[i].frac_coords for i in indices]
        shuffled_structure = Structure(structure.lattice, new_species, new_coords)
        return shuffled_structure, indices
    
    @staticmethod
    def calculate_cosine_angle(v1, v2):
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1, 1) #adjust wrong values due to limited precision
        return cosine_angle


    def shuffle_structure_and_get_graph(self,crystal_data):
        """
        shuffle structures, generate atom graphs in real time.
        dimensions:
        atom_num:[n,1]; nbr_fea:[n,num_nbr,41]; nbr_fea_idx:[n,12]; distance_matrix:[n,n]; atom_to_molecule_list: [n,1]; 
        angles_list:[n,n_nbr*(n_nbr-1)/2].
        """
        structure=crystal_data["structure"]
        cif_id = crystal_data["cif_id"]
        original_atom_num = crystal_data["atom_num"]

        # # if if_alignn, do not shuffle
        # if not self.if_alignn:
        #     shuffled_structure, indices = self.shuffle_structure(structure)
        #     graph=None
        #     line_graph=None
        # else:
        #     shuffled_structure=structure
        #     indices=list(range(len(structure)))
        #     graph=crystal_data["graph"]
        #     line_graph=crystal_data["line_graph"]

        # only when train, shuffle the structure
        if self.split=="train":
            shuffled_structure, indices = self.shuffle_structure(structure)
        else:
            shuffled_structure=structure
            indices=list(range(len(structure)))

        if not self.if_alignn:
            graph=None
            line_graph=None
        else:
            atoms = convert_structures_to_jarvis_atoms(shuffled_structure)
            graph = jarvis_atoms_to_dgl_graph(
                        atoms,
                        "k-nearest",
                        8.0,
                        12,
                        True,
                    )
            graph.apply_edges(
                        lambda edges: {"distance": torch.norm(edges.data["coord_diff"], dim=1)}
                    )
            line_graph = graph.line_graph(shared=True)
            line_graph.apply_edges(compute_bond_cosines)

        # re-order atom_num
        atom_num = [original_atom_num[i] for i in indices]
        atom_num = torch.LongTensor(atom_num)

        #atom graph for cgcnn
        if self.if_conv:
            nbr_fea, nbr_fea_idx=self.get_atom_graph(structure=shuffled_structure,radius=8,max_num_nbr=self.max_num_nbr)
        else:
            nbr_fea=None
            nbr_fea_idx=None

        #distance_matrix
        distance_matrix=shuffled_structure.distance_matrix
        distance_matrix = torch.Tensor(distance_matrix)

        # positional embedding
        if self.pos_emb=='relative':
            original_angles_list = crystal_data["angles_list"]
            angles_list = [original_angles_list[i] for i in indices]
            angles_list = torch.Tensor(angles_list)
            abs_pos=None
        elif self.pos_emb=='absolute':
            abs_pos=shuffled_structure.cart_coords
            abs_pos=torch.Tensor(abs_pos)
            angles_list=None
        elif self.pos_emb=='both':
            original_angles_list = crystal_data["angles_list"]
            angles_list = [original_angles_list[i] for i in indices]
            angles_list = torch.Tensor(angles_list)
            abs_pos=shuffled_structure.cart_coords
            abs_pos=torch.Tensor(abs_pos)
        else:
            angles_list=None
            abs_pos=None
            pass

        # # mask atoms
        # if "map" in self.tasks:
        #     atom_num, atom_label = self.apply_mask_to_atom_num(atom_num, mask_probability=self.mask_probability)
        # else:
        #     atom_label = torch.full_like(atom_num, -1)

        # randomly select atom pairs and generate labels using atm_list
        if "apc" in self.tasks:
            original_atm_list = crystal_data["atm_list"]
            atm_list = [original_atm_list[i] for i in indices]
            atom_pairs, ap_labels = self.select_atom_pairs(atm_list, self.num_ap) #  [m, 2],[m]
        else:        
            atom_pairs = None
            ap_labels = None

        if "adp" in self.tasks:
            dist_pairs, dist_labels = self.select_dist_pairs(distance_matrix, self.n_dist)
        else:
            dist_pairs = None
            dist_labels = None

        if "aap" in self.tasks:
            angle_pairs, angle_labels = self.select_angle_pairs(shuffled_structure, self.n_angle)
        else:
            angle_pairs = None
            angle_labels = None

        if "ucp" in self.tasks:
            lattice = shuffled_structure.lattice
            lengths = np.array(lattice.lengths)
            angles = np.array(lattice.angles) 

            length_norm = 60.0  # normalize
            angle_norm = 180.0  # normalize
            norm_lengths = np.round(lengths / length_norm, 4)
            norm_angles = np.round(angles / angle_norm, 4)
            
            ucp_labels = list(norm_lengths) + list(norm_angles)  # [6]
        else:
            ucp_labels = None

        crystal_data={
                    "atom_num":atom_num,
                    "nbr_fea":nbr_fea, 
                    "nbr_fea_idx":nbr_fea_idx,
                    "distance_matrix":distance_matrix,
                    # "atm_list":atm_list,
                    "atom_pairs":atom_pairs,
                    "ap_labels":ap_labels,
                    "dist_pairs":dist_pairs,
                    "dist_labels":dist_labels,
                    "angle_pairs":angle_pairs,
                    "angle_labels":angle_labels,
                    "ucp_labels":ucp_labels,
                    "angles_list":angles_list,
                    "abs_pos":abs_pos,
                    "graph":graph,
                    "line_graph":line_graph,
                    "cif_id":cif_id,
                }
        return crystal_data

    @staticmethod
    def read_structure(file_path):
        try:
            structure = Structure.from_file(file_path, occupancy_tolerance=100.0)
            return structure
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    @staticmethod
    def symmetry_operation(structure):
        # Redirect stdout and stderr to /dev/null
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            analyzer = SpacegroupAnalyzer(structure)
            symmetrized_structure = analyzer.get_symmetrized_structure()
            return symmetrized_structure
        except Exception as e:
            print(f"Error during symmetry operation: {e}", file=sys.stderr)
    

    @staticmethod
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
            if len(all_distances_sorted) < n_nbr:
                n_nbr_distance = all_distances_sorted[-1]  # Use the last available distance if less than n_nbr
                # Directly select list of nearest neighbors
                nearest_neighbors_indices = all_indices_sorted
                nearest_distances = all_distances_sorted
            else:
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

    def get_graph(self, cif_id):
        if self.read_from_pickle:
            file_structure = os.path.join(self.pickle_path, f"{cif_id}.pickle")
            with open(file_structure, "rb") as f:
                crystal_data = pickle.load(f)
        else:#generate in real time
            file_cif = os.path.join(self.cif_path, f"{cif_id}.cif")
            structure=self.read_structure(file_cif)
            structure=self.symmetry_operation(structure)
            if structure.num_sites > 1023:
                print("cif",cif_id,": Atom length {structure.num_sites} > 1023, please remove it, will not build its graph")
                return None
            else:
                atom_num = [site.specie.Z for site in structure]
                if self.pos_emb=='relative' or self.pos_emb=='both':
                    angles_list= self.get_nearest_neighbors_cos_angles(structure=structure, n_nbr=8)
                else:
                    angles_list=None
                    
                crystal_data={
                    "structure":structure,
                    "atom_num":atom_num,
                    "angles_list":angles_list,
                    "cif_id":cif_id,
                }


        graphdata = self.shuffle_structure_and_get_graph(crystal_data)
        # expand nbr_fea
        if self.if_conv:
            graphdata["nbr_fea"]=self.get_gaussian_distance(graphdata["nbr_fea"], num_step=self.nbr_fea_len).float()
        # # get image
        # if self.if_image:
        #     image_path = os.path.join(self.image_path, f"{cif_id}.npy")
        #     image_data  = np.load(image_path, allow_pickle=True)
        #     # Extract the image array and numerical values
        #     image = image_data[0]
        #     max_birth_1d = float(image_data[1])
        #     max_persistence_1d = float(image_data[2])
        #     max_birth_2d = float(image_data[3])
        #     max_persistence_2d = float(image_data[4])
        #     if image.shape[0] == 3:
        #         graphdata["image"] = torch.from_numpy(image[1:, :, :]).float()
        #     else:
        #         graphdata["image"] = torch.from_numpy(image).float()
        #     # Creating tensors from scalar values
        #     graphdata["max_birth_1d"] = max_birth_1d
        #     graphdata["max_persistence_1d"] = max_persistence_1d
        #     graphdata["max_birth_2d"] = max_birth_2d
        #     graphdata["max_persistence_2d"] = max_persistence_2d
        return graphdata  
    
    def get_image(self, cif_id):
        image_path = os.path.join(self.image_path, f"{cif_id}.npy")
        image_data  = np.load(image_path, allow_pickle=True)
        # Extract the image array and numerical values
        image = image_data[0]
        max_birth_1d = float(image_data[1])
        max_persistence_1d = float(image_data[2])
        max_birth_2d = float(image_data[3])
        max_persistence_2d = float(image_data[4])

        imagedata = dict()
        if image.shape[0] == 3:
            imagedata["image"] = torch.from_numpy(image[1:, :, :]).float()
        else:
            imagedata["image"] = torch.from_numpy(image).float()
        # Creating tensors from scalar values
        imagedata["max_birth_1d"] = max_birth_1d
        imagedata["max_persistence_1d"] = max_persistence_1d
        imagedata["max_birth_2d"] = max_birth_2d
        imagedata["max_persistence_2d"] = max_persistence_2d
        return imagedata      

    @staticmethod
    def make_grid_data(grid_data, emin=-5000.0, emax=5000, bins=101):
        """
        make grid_data within range (emin, emax) and
        make bins with logit function
        and digitize (0, bins)
        ****
            caution : 'zero' should be padding !!
            when you change bins, heads.MPP_heads should be changed
        ****
        """
        grid_data[grid_data <= emin] = emin
        grid_data[grid_data > emax] = emax

        x = np.linspace(emin, emax, bins)
        new_grid_data = np.digitize(grid_data, x) + 1

        return new_grid_data

    @staticmethod
    def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
        a_ = np.cos(angle_a * np.pi / 180)
        b_ = np.cos(angle_b * np.pi / 180)
        c_ = np.cos(angle_c * np.pi / 180)

        v = a * b * c * np.sqrt(1 - a_**2 - b_**2 - c_**2 + 2 * a_ * b_ * c_)

        return v.item() / (60 * 60 * 60)  # normalized volume

    def get_raw_grid_data(self, cif_id):
        file_grid = os.path.join(self.grid_path, f"{cif_id}.grid")
        file_griddata = os.path.join(self.grid_path, f"{cif_id}.griddata16")

        # get grid
        with open(file_grid, "r") as f:
            lines = f.readlines()
            a, b, c = [float(i) for i in lines[0].split()[1:]]
            angle_a, angle_b, angle_c = [float(i) for i in lines[1].split()[1:]]
            cell = [int(i) for i in lines[2].split()[1:]]

        volume = self.calculate_volume(a, b, c, angle_a, angle_b, angle_c)

        # get grid data
        grid_data = pickle.load(open(file_griddata, "rb"))
        grid_data = self.make_grid_data(grid_data)
        grid_data = torch.FloatTensor(grid_data)

        return cell, volume, grid_data

    def get_grid_data(self, cif_id, draw_false_grid=False):
        cell, volume, grid_data = self.get_raw_grid_data(cif_id)
        ret = {
            "cell": cell,
            "volume": volume,
            "grid_data": grid_data,
        }

        if draw_false_grid:
            random_index = random.randint(0, len(self.cif_ids) - 1)
            cif_id = self.cif_ids[random_index]
            cell, volume, grid_data = self.get_raw_grid_data(cif_id)
            ret.update(
                {
                    "false_cell": cell,
                    "fale_volume": volume,
                    "false_grid_data": grid_data,
                }
            )
        return ret   


    def __getitem__(self, index):
        crystal_data = dict()
        cif_id = self.cif_ids[index]

        temp_graph_data = self.get_graph(cif_id)
        crystal_data.update(temp_graph_data if temp_graph_data is not None else print(f"{cif_id} has more than 1023 atoms, please use cif_to_data.py to rule out structures with more than 1023 atoms")) #get graph data and other labels

        if self.if_image:
            crystal_data.update(self.get_image(cif_id)) # get image data
        if self.if_grid:
            crystal_data.update(self.get_grid_data(cif_id)) # get grid data


        if self.downstream:
            downstream_data=self.id_downstream_data[cif_id]
            crystal_data.update({"target": downstream_data,})
        
        if "cdp" in self.tasks:
            density = float(self.id_density_data[cif_id])
            crystal_data.update({"density": density,})
            
        if "sgp" in self.tasks:    
            space_group = int(self.id_space_group_data[cif_id])
            crystal_data.update({"space_group": space_group,})
        
        if "sep" in self.tasks:    
            space_group = self.id_space_group_data[cif_id]
            symm_elem = self.symm_elem_data[space_group]
            crystal_data.update({"symm_elem": symm_elem,})
        """
        crystal_data = {
            "atom_num":atom_num,                           (torch.Tensor),     [n,1]
            "nbr_fea":nbr_fea,                             (torch.Tensor),     [n,num_nbr,self.nbr_fea_len]
            "nbr_fea_idx":nbr_fea_idx,                     (torch.LongTensor), [n,12]
            "distance_matrix":distance_matrix,             (torch.Tensor),     [n,n]
            "angles_list":angles_list,                     (torch.Tensor),     [n,n_nbr*(n_nbr-1)/2]
            "cif_id":cif_id,                               (str),  
            "density"                                      (float),
            "space_group"                                  (int),
            "symm_elem"                                    (list),             [22]
            / {downstream}                               }
        """

        return crystal_data

    @staticmethod
    def collate(batch,max_graph_len: int=1023,angle_nbr:int=8,mask_probability:float=0.15,tasks=[],if_conv=False,pos_emb="relative",if_alignn=True):
        """
        collate batch
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])

        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        # image
        if 'image' in dict_batch:
            dict_batch["image"] = torch.stack(dict_batch["image"], dim=0)
            # dict_batch["max_birth_1d"] = torch.stack(dict_batch["max_birth_1d"], dim=0)
            # dict_batch["max_persistence_1d"] = torch.stack(dict_batch["max_persistence_1d"], dim=0)
            # dict_batch["max_birth_2d"] = torch.stack(dict_batch["max_birth_2d"], dim=0)
            # dict_batch["max_persistence_2d"] = torch.stack(dict_batch["max_persistence_2d"], dim=0)
        if "grid_data" in dict_batch:
            from torch.nn.functional import interpolate
            batch_grid_data = dict_batch["grid_data"]
            batch_cell = dict_batch["cell"]
            new_grids = []

            for bi in range(batch_size):
                orig = batch_grid_data[bi].view(batch_cell[bi][::-1]).transpose(0, 2)
                if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                    orig = orig[None, None, :, :, :]
                else:
                    orig = interpolate(
                        orig[None, None, :, :, :],
                        size=[30, 30, 30],
                        mode="trilinear",
                        align_corners=True,
                    )
                new_grids.append(orig)
            new_grids = torch.concat(new_grids, axis=0)
            dict_batch["grid"] = new_grids

            if "false_grid_data" in dict_batch.keys():
                batch_false_grid_data = dict_batch["false_grid_data"]
                batch_false_cell = dict_batch["false_cell"]
                new_false_grids = []
                for bi in range(batch_size):
                    orig = batch_false_grid_data[bi].view(batch_false_cell[bi])
                    if batch_cell[bi] == [30, 30, 30]:  # version >= 1.1.2
                        orig = orig[None, None, :, :, :]
                    else:
                        orig = interpolate(
                            orig[None, None, :, :, :],
                            size=[30, 30, 30],
                            mode="trilinear",
                            align_corners=True,
                        )
                    new_false_grids.append(orig)
                new_false_grids = torch.concat(new_false_grids, axis=0)
                dict_batch["false_grid"] = new_false_grids

            dict_batch.pop("grid_data", None)
            dict_batch.pop("false_grid_data", None)
            dict_batch.pop("cell", None)
            dict_batch.pop("false_cell", None)            

        #atom graph for cgcnn
        batch_atom_num = dict_batch["atom_num"]

        #for cgcnn
        if if_conv:
            batch_nbr_idx = dict_batch["nbr_fea_idx"]
            batch_nbr_fea = dict_batch["nbr_fea"]
            crystal_atom_idx = []
            base_idx = 0
            for i, nbr_idx in enumerate(batch_nbr_idx):
                n_i = nbr_idx.shape[0]
                crystal_atom_idx.append(torch.arange(n_i) + base_idx)
                nbr_idx += base_idx
                base_idx += n_i
            dict_batch["nbr_fea_idx"] = torch.cat(batch_nbr_idx, dim=0)
            dict_batch["nbr_fea"] = torch.cat(batch_nbr_fea, dim=0)
        else:
            crystal_atom_idx = []
            base_idx = 0
            for i, nbr_idx in enumerate(batch_atom_num):
                n_i = nbr_idx.shape[0]
                crystal_atom_idx.append(torch.arange(n_i) + base_idx)
                base_idx += n_i

        dict_batch["atom_num"] = torch.cat(batch_atom_num, dim=0)
        dict_batch["crystal_atom_idx"] = crystal_atom_idx

        #dgl graph 
        if if_alignn:
            dict_batch["graph"] = dgl.batch(dict_batch["graph"])
            dict_batch["line_graph"] = dgl.batch(dict_batch["line_graph"])
            
        #for apc task
        if "apc" in tasks:
            batch_atom_pairs = dict_batch["atom_pairs"]
            batch_ap_labels = dict_batch["ap_labels"]
            dict_batch["atom_pairs"] = torch.stack(batch_atom_pairs)  # [B, m, 2]
            dict_batch["ap_labels"] = torch.stack(batch_ap_labels)    # [B, m]

        #for adp task
        if "adp" in tasks:
            batch_dist_pairs = dict_batch["dist_pairs"]
            batch_dist_labels = dict_batch["dist_labels"]
            dict_batch["dist_pairs"] = torch.stack(batch_dist_pairs)  # [B, n_dist, 2]
            dict_batch["dist_labels"] = torch.stack(batch_dist_labels)    # [B, n_dist]

        #for aap task
        if "aap" in tasks:
            batch_angle_pairs = dict_batch["angle_pairs"]
            batch_angle_labels = dict_batch["angle_labels"]
            dict_batch["angle_pairs"] = torch.stack(batch_angle_pairs)  # [B, n_angle, 2]
            dict_batch["angle_labels"] = torch.stack(batch_angle_labels)    # [B, n_angle]

        #for map task
        # mask atoms
        def apply_mask_to_atom_num(atom_num, mask_probability=0.15):
            """
            Applies masking logic directly to atom numbers in the dataset.
            Returns:
            masked_atom_num (torch.Tensor): Tensor of atom numbers after applying mask.
            atom_label (torch.Tensor): Labels tensor for loss computation, with -1 indicating atoms to ignore.
            """
            atom_label = torch.full_like(atom_num, -1)
            mask = torch.rand(atom_num.size(0)) < mask_probability
            masked_atom_num = atom_num.clone()

            for i in range(masked_atom_num.size(0)):
                if mask[i]:
                    atom_label[i] = atom_num[i]# Set the correct label for masked atom
                    decision = torch.rand(1).item()
                    if decision < 0.8:  # 80% chance to replace with [MASK], in this case 0
                        masked_atom_num[i] = 0
                    elif decision < 0.9:  # 10% chance for random replacement
                        masked_atom_num[i] = torch.randint(1, 119, (1,)).item()
                    # Remaining 10% chance to keep unchanged
                    
            return masked_atom_num, atom_label 
        if "map" in tasks:
            masked_atom_num, atom_label = apply_mask_to_atom_num(dict_batch["atom_num"],mask_probability)
            dict_batch["atom_num"]=masked_atom_num
            padded_atom_label = torch.full((batch_size, max_graph_len), -1)
            for i, idx in enumerate(crystal_atom_idx):
                    length = len(idx)
                    padded_atom_label[i, :length] = atom_label[idx]
            dict_batch["atom_label"]=padded_atom_label
        
        # for positional embedding
        if pos_emb == "relative":
            #distance
            batch_distance_matrix = dict_batch["distance_matrix"]
            padded_distance_matrices = torch.zeros(batch_size, max_graph_len, max_graph_len)
            mask_distance_matrices = torch.zeros(batch_size, max_graph_len, max_graph_len)
            for i, matrix in enumerate(batch_distance_matrix):
                n = matrix.size(0)
                padded_distance_matrices[i, :n, :n] = matrix
                mask_distance_matrices[i, :n, :n] = 1
            dict_batch["padded_distance_matrices"] = padded_distance_matrices
            dict_batch["mask_distance_matrices"] = mask_distance_matrices

            #angle
            def generate_indices(angle_nbr):
                indices = []
                start = 0
                for i in range(7, 0, -1):
                    end = start + i
                    indices.extend(range(start, start + angle_nbr-1))
                    start = end
                    angle_nbr -= 1
                    if angle_nbr <= 0:
                        break
                return indices
            #select angles based on angle_nbr
            batch_angles_list= dict_batch["angles_list"] 
            angle_indices=generate_indices(angle_nbr)
            for i, tensor in enumerate(batch_angles_list):
                batch_angles_list[i]=tensor[:, angle_indices]

            x_dim = batch_angles_list[0].shape[1]
            padded_angle_matrices = torch.full((batch_size, max_graph_len, x_dim), -100.0)
            mask_angle_matrices = torch.full((batch_size, max_graph_len, x_dim), 1)
            for i, matrix in enumerate(batch_angles_list):
                n = matrix.shape[0]
                padded_angle_matrices[i, :n, :] = matrix

            mask_angle_matrices[padded_angle_matrices == -100.0] = 0
            dict_batch["padded_angle_matrices"] = padded_angle_matrices
            dict_batch["mask_angle_matrices"] = mask_angle_matrices   

        elif pos_emb == "absolute":
            batch_abs_pos = dict_batch["abs_pos"]
            pos_dim = batch_abs_pos[0].shape[1]
            padded_abs_pos = torch.zeros(batch_size, max_graph_len, pos_dim)
            for i, matrix in enumerate(batch_abs_pos):
                n = matrix.shape[0]
                padded_abs_pos[i, :n, :] = matrix
            dict_batch["padded_abs_pos"] = padded_abs_pos

        elif pos_emb=='both':
            #distance
            batch_distance_matrix = dict_batch["distance_matrix"]
            padded_distance_matrices = torch.zeros(batch_size, max_graph_len, max_graph_len)
            mask_distance_matrices = torch.zeros(batch_size, max_graph_len, max_graph_len)
            for i, matrix in enumerate(batch_distance_matrix):
                n = matrix.size(0)
                padded_distance_matrices[i, :n, :n] = matrix
                mask_distance_matrices[i, :n, :n] = 1
            dict_batch["padded_distance_matrices"] = padded_distance_matrices
            dict_batch["mask_distance_matrices"] = mask_distance_matrices

            #angle
            def generate_indices(angle_nbr):
                indices = []
                start = 0
                for i in range(7, 0, -1):
                    end = start + i
                    indices.extend(range(start, start + angle_nbr-1))
                    start = end
                    angle_nbr -= 1
                    if angle_nbr <= 0:
                        break
                return indices
            #select angles based on angle_nbr
            batch_angles_list= dict_batch["angles_list"] 
            angle_indices=generate_indices(angle_nbr)
            for i, tensor in enumerate(batch_angles_list):
                batch_angles_list[i]=tensor[:, angle_indices]

            x_dim = batch_angles_list[0].shape[1]
            padded_angle_matrices = torch.full((batch_size, max_graph_len, x_dim), -100.0)
            mask_angle_matrices = torch.full((batch_size, max_graph_len, x_dim), 1)
            for i, matrix in enumerate(batch_angles_list):
                n = matrix.shape[0]
                padded_angle_matrices[i, :n, :] = matrix

            mask_angle_matrices[padded_angle_matrices == -100.0] = 0
            dict_batch["padded_angle_matrices"] = padded_angle_matrices
            dict_batch["mask_angle_matrices"] = mask_angle_matrices  
            #absolute
            batch_abs_pos = dict_batch["abs_pos"]
            pos_dim = batch_abs_pos[0].shape[1]
            padded_abs_pos = torch.zeros(batch_size, max_graph_len, pos_dim)
            for i, matrix in enumerate(batch_abs_pos):
                n = matrix.shape[0]
                padded_abs_pos[i, :n, :] = matrix
            dict_batch["padded_abs_pos"] = padded_abs_pos
        else:
            pass

        return dict_batch
