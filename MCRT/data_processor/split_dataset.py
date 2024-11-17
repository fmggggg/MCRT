import os
import random
import argparse
import json

def load_existing_split(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_split(output_path, dataset_split):
    with open(output_path, 'w') as f:
        json.dump(dataset_split, f, indent=4)

def split_data(input_folder, split_ratio, random_seed, train_count, val_count, split_json):
    random.seed(random_seed)
    cif_files = [file[:-4] for file in os.listdir(input_folder) if file.endswith('.cif')]

    if split_json:
        dataset_split = load_existing_split(split_json)
    else:
        dataset_split = {'train': [], 'val': [], 'test': cif_files[:]}

    random.shuffle(dataset_split['test'])

    if train_count is not None and val_count is not None:
        total_needed = train_count + val_count
        existing_train_count = len(dataset_split['train'])
        existing_val_count = len(dataset_split['val'])
        
        if total_needed > len(cif_files):
            raise ValueError(f"Not enough files to meet the requested train ({train_count}) and val ({val_count}) count.")
        
        train_needed = max(0, train_count - existing_train_count)
        val_needed = max(0, val_count - existing_val_count)

        if train_needed > 0:
            dataset_split['train'].extend(dataset_split['test'][:train_needed])
        if val_needed > 0:
            dataset_split['val'].extend(dataset_split['test'][train_needed:train_needed + val_needed])

        # Update the test set to remove any files that have been added to train or val
        remaining_test_files = dataset_split['test'][train_needed + val_needed:]
        dataset_split['test'] = [file for file in remaining_test_files if file not in dataset_split['train'] and file not in dataset_split['val']]
    else:
        total_files = len(dataset_split['test'])
        train_size = int(split_ratio[0] * total_files)
        val_size = int(split_ratio[1] * total_files)
        test_size = total_files - train_size - val_size

        dataset_split['train'].extend(dataset_split['test'][:train_size])
        dataset_split['val'].extend(dataset_split['test'][train_size:train_size + val_size])
        dataset_split['test'] = dataset_split['test'][train_size + val_size:train_size + val_size + test_size]

    save_split(os.path.join(os.path.dirname(input_folder), 'dataset_split.json'), dataset_split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CIF files into train, val, and test sets.")
    parser.add_argument("--cif", type=str, required=True, help="Path to the input folder containing CIF files.")
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="List containing the ratio of CIF files to be used for training, validation, and test sets.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for shuffling the dataset.")
    parser.add_argument("--train_count", type=int, help="Number of CIF files to be used for the training set.")
    parser.add_argument("--val_count", type=int, help="Number of CIF files to be used for the validation set.")
    parser.add_argument("--split_json", type=str, help="Path to an existing JSON file with dataset split information.")
    args = parser.parse_args()

    # Check if split ratios sum to 1
    if sum(args.split) != 1:
        raise ValueError("The sum of split ratios must be 1.")

    split_data(args.cif, args.split, args.seed, args.train_count, args.val_count, args.split_json)
