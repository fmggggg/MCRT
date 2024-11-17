import warnings
warnings.filterwarnings("ignore")
from pymatgen.core.structure import Structure
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

from GNN.datasets.utils_jarvis import (
    jarvis_atoms_to_dgl_graph,
    compute_bond_cosines,
    convert_structures_to_jarvis_atoms,
)

def main():
    parser = argparse.ArgumentParser(description='cif_path')
    parser.add_argument('--cif_path', type=str, help='input cif_path, the output will in this file too')
    args = parser.parse_args()
    if args.cif_path:
        cif_path = args.cif_path
    else:
        raise KeyError("No cif_path found")
    # Check if the source folder exists
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"Source folder not found at {cif_path}")
    
    cif_files = [os.path.join(cif_path, f) for f in os.listdir(cif_path) if f.endswith('.cif')]
    print('prepared cifs list')
    for cif_file in tqdm(cif_files, desc="Identifying molecules"):
            parse_cif(cif_file)
    print('End parse cifs pickles' )




def read_structure(file_path):
    try:
        structure = Structure.from_file(file_path, occupancy_tolerance=100.0)
        # name = os.path.splitext(os.path.basename(file_path))[0]
        return structure, file_path
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


def identify_and_label_molecules(symmetrized_structure_with_name):
    """
    Identify and label molecules in a crystal structure using StructureGraph.
    """
    structure, file_path = symmetrized_structure_with_name
    cif_id = os.path.splitext(os.path.basename(file_path))[0]
    atoms = convert_structures_to_jarvis_atoms(structure)
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
        
    crystal_data={
                "structure":structure,
                "graph":graph,
                "line_graph":line_graph,
                "cif_id":cif_id,
            }
    pickle_path=os.path.join(os.path.dirname(file_path),"pickles")
    if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
    with open(os.path.join(pickle_path, f"{cif_id}.pickle"), "wb") as file:
            pickle.dump(crystal_data, file) 

      

def parse_cif(file_path):
    try:
        structures_with_names=read_structure(file_path)
        identify_and_label_molecules(structures_with_names)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        pass
    
def parse_cifs_in_parallel(file_paths, max_workers):
    """
    Process multiple structures in parallel using ProcessPoolExecutor.
    """
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dataset = list(tqdm(executor.map(parse_cif, file_paths), total=len(file_paths), desc="Identifying molecules in parallel"))

    end_time = time.time()
    elapsed_time = end_time - start_time
    structures_per_second = len(file_paths) / elapsed_time

    print(f"identify_molecules_Processed {len(file_paths)} structures in {elapsed_time:.2f} seconds.")
    print(f"identify_molecules_Average speed: {structures_per_second:.2f} structures/second")
    return dataset

if __name__ == "__main__":
    main()
