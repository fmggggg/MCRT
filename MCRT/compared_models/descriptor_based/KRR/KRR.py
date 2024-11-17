import os
import argparse
import numpy as np
import pandas as pd
import pickle
import copy
import json
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Modify argparse parameters
parser = argparse.ArgumentParser(description="""descriptor_based RF""")
parser.add_argument('--descriptor-csv', type=str,default=r'D:\Projects\MyProjects\MCRT\MCRT\compared_models\descriptor_based\triptycene_soap_descriptors.csv', help='(str) Path to the descriptor CSV file.')
parser.add_argument('--label-csv', type=str,default=r'D:\Projects\MyProjects\MCRT\MCRT\compared_models\descriptor_based\Triptycene_energy.csv', help='(str) Path to the label CSV file.')
parser.add_argument('--output', '-o', type=str, default=None, help='(str) Output file path')
parser.add_argument('--outdir', type=str, default=None, help='(str) Directory to save the model')
parser.add_argument('--load-model', type=str, default=None, help='(str) Path to load a pre-trained model')
parser.add_argument('--results_csv', type=str, default=r'D:\projects\MCRT\MCRT\compared_models\descriptor_based\KRR\KRR_T2_diffusity_results.csv', help='(str) Path to save the test results')
parser.add_argument('--split', type=str, default=r'D:\Projects\MyProjects\MCRT\MCRT\cifs\Triptycene_ALL_CIF\dataset_split.json', help='(str) Path to the JSON file with train/val/test split information')
args = parser.parse_args()

def get_data(descriptor_csv, label_csv):
    # Read descriptor and label data without headers
    df_descriptors = pd.read_csv(descriptor_csv, header=None, index_col=0)
    df_labels = pd.read_csv(label_csv, header=None, index_col=0)

    # Match descriptors and labels by material name
    df = df_descriptors.join(df_labels, how='inner', rsuffix='_label')
    
    # Separate features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(X.shape)
    return X, y

def main():
    X, y = get_data(args.descriptor_csv, args.label_csv)
    
    # Load the predefined split from JSON file
    with open(args.split, 'r') as f:
        split_info = json.load(f)
    
    # Get the indices for train, val, and test sets
    train_indices = [idx for idx in split_info['train'] if idx in y.index]
    val_indices = [idx for idx in split_info['val'] if idx in y.index]
    test_indices = [idx for idx in split_info['test'] if idx in y.index]
    
    # Split the data according to the predefined indices
    X_train = X.loc[train_indices].values
    y_train = y.loc[train_indices].values
    X_val = X.loc[val_indices].values
    y_val = y.loc[val_indices].values
    X_test = X.loc[test_indices].values
    y_test = y.loc[test_indices].values
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    best_score = -np.inf
    best_model = None
    best_alpha = 0

    if args.load_model:
        if not os.path.exists(args.load_model):
            raise ValueError('model does not exists!')

        with open(args.load_model, 'rb') as f:
            load_model = pickle.load(f)
        best_model = load_model
        # Predict on the test data
        y_pred = load_model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R^2 Score: {r2}")

        # Save actual and predicted values to a CSV file
        results_df = pd.DataFrame({'ID': y.loc[test_indices].index, 'Actual': y_test, 'Predicted': y_test_pred})
        results_df.to_csv(args.results_csv, index=False)
        print(f"Actual and predicted values have been saved to {args.results_csv}")

        # Plotting
        plot_path = os.path.splitext(args.results_csv)[0] + '_krr.png'
        x = np.array(y_pred)
        y = np.array(y_test)
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        z = (z - z.min()) / (z.max() - z.min())
        fig, ax = plt.subplots()
        scatter = ax.scatter(y, x, c=z, s=0.5, cmap='rainbow')
        plt.colorbar(scatter, label='Point Density')
        plt.title("Prediction vs. Actual")
        plt.xlabel('Actual Values') 
        plt.ylabel('Predicted Values')
        min_val, max_val = min(min(y), min(x)), max(max(y), max(x))
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, linestyle="--", linewidth=1, color="black")
        plt.savefig(plot_path, format="png")
        plt.show()

    else:
        for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:

            kernel = 'rbf'  
            gamma = None 

            model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma)
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)

            if args.outdir:
                os.makedirs(args.outdir, exist_ok=True)
                with open(os.path.join(args.outdir, f'KRR_{alpha}.pickle'), 'wb') as f:
                    pickle.dump(model, f)
            print(f'finish {alpha} r2_score:{score} mae: {mae}')  
            
            if score > best_score:
                best_score = score
                best_model = copy.deepcopy(model)
                best_alpha = alpha  
            print(f"Best R^2 Score: {best_score} with alpha: {best_alpha}")

        # Predict on the test set with the best model
        y_test_pred = best_model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"Test set evaluation: MAE = {test_mae}, R^2 Score = {test_r2}")

        # Save actual and predicted values to a CSV file
        results_df = pd.DataFrame({'ID': y.loc[test_indices].index, 'Actual': y_test, 'Predicted': y_test_pred})
        results_df.to_csv(args.results_csv, index=False)
        print(f"Actual and predicted values have been saved to {args.results_csv}")

    # Save the best model
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, 'best_model.pickle'), 'wb') as f:
            pickle.dump(best_model, f)
    
    # Write the results to the output file
    if args.output:
        with open(args.output, 'a') as f:
            f.write(f"Best R^2 Score: {best_score} with n_estimators: {best_alpha}\n")

if __name__ == '__main__':
    main()
