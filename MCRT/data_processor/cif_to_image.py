import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import time
import numpy as np
import pickle
import argparse
from moleculetda.structure_to_vectorization import structure_to_pd
from moleculetda.vectorize_pds import pd_vectorization

def main():
    parser = argparse.ArgumentParser(description='cif_path')
    parser.add_argument('--cif_path', type=str, required=True, help='Input CIF path. The output will be in this folder too.')
    parser.add_argument('--paral', type=int, help='If parallel, define number of max_workers (int, >= 2).')
    parser.add_argument('--save_dgm', type=str, default='False', help='If True, save the diagram as a pickle file. Must be "True" or "False".')
    args = parser.parse_args()

    cif_path = args.cif_path
    paral = args.paral if args.paral and args.paral >= 2 else False

    if args.save_dgm not in ['True', 'False']:
        raise ValueError('The save_dgm parameter must be "True" or "False".')
    save_dgm = args.save_dgm == 'True'

    # Check if the source folder exists
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"Source folder not found at {cif_path}")
    
    cif_files = [os.path.join(cif_path, f) for f in os.listdir(cif_path) if f.endswith('.cif')]
    npy_path = os.path.join(cif_path, "imgs")
    if os.path.exists(npy_path):
        npy_files = [os.path.join(cif_path, f) for f in os.listdir(npy_path) if f.endswith('.npy')]
        npy_names = set(os.path.splitext(os.path.basename(f))[0] for f in npy_files)
        cif_files = [f for f in cif_files if os.path.splitext(os.path.basename(f))[0] not in npy_names]
    print('Prepared CIFs list')
    
    if not paral:
        for cif_file in tqdm(cif_files, desc="Identifying images"):
            parse_cif(cif_file, save_dgm)
    else:
        parse_cifs_in_parallel(cif_files, max_workers=paral, save_dgm=save_dgm)
    
    print('End parse CIFs pickles')

def parse_cif(file_path, save_dgm):
    try:
        cif_id = os.path.splitext(os.path.basename(file_path))[0]
        arr_dgms = structure_to_pd(file_path, supercell_size=100)
        dgm_1d = arr_dgms['dim1']
        dgm_2d = arr_dgms['dim2']
        image_1d = pd_vectorization(dgm_1d, spread=0.15, weighting='identity', pixels=[50, 50])
        image_2d = pd_vectorization(dgm_2d, spread=0.15, weighting='identity', pixels=[50, 50])
        image = np.stack([image_1d, image_2d])
        max_birth_1d = np.max(dgm_1d["birth"]) if dgm_1d["birth"].size > 0 else 0
        max_persistence_1d = np.max(dgm_1d["death"] - dgm_1d["birth"]) if dgm_1d["birth"].size > 0 else 0
        max_birth_2d = np.max(dgm_2d["birth"]) if dgm_2d["birth"].size > 0 else 0
        max_persistence_2d = np.max(dgm_2d["death"] - dgm_2d["birth"]) if dgm_2d["birth"].size > 0 else 0

        # Prepare directory to save the outputs
        npy_path = os.path.join(os.path.dirname(file_path), "imgs")
        if not os.path.exists(npy_path):
            os.makedirs(npy_path)

        # Save numerical values and images together
        values_and_images = np.array([image, max_birth_1d, max_persistence_1d, max_birth_2d, max_persistence_2d], dtype=object)
        file_to_save_npy = os.path.join(npy_path, f"{cif_id}.npy")
        np.save(file_to_save_npy, values_and_images)

        if save_dgm:
            # Save arr_dgms as a separate pickle file for space efficiency
            dgm_path = os.path.join(os.path.dirname(file_path), "dgms")
            if not os.path.exists(dgm_path):
                os.makedirs(dgm_path)
            file_to_save_pickle = os.path.join(dgm_path, f"{cif_id}_dgms.pickle")
            with open(file_to_save_pickle, 'wb') as f:
                pickle.dump(arr_dgms, f)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

def parse_cifs_in_parallel(file_paths, max_workers, save_dgm):
    """
    Process multiple structures in parallel using ProcessPoolExecutor.
    """
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(parse_cif, file_path, save_dgm): file_path for file_path in file_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Identifying images in parallel"):
            future.result()

    end_time = time.time()
    elapsed_time = end_time - start_time
    structures_per_second = len(file_paths) / elapsed_time

    print(f"Processed {len(file_paths)} structures in {elapsed_time:.2f} seconds.")
    print(f"Average speed: {structures_per_second:.2f} structures/second")

if __name__ == "__main__":
    main()
