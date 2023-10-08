import importlib
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

def load_parameters(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def save_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def cluster_data(class_name, params_path, fit_params_path, scaling_option, input_csv, output_dir):
    # Dynamically import clustering class
    module_name, class_name = class_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    clustering_class = getattr(module, class_name)

    # Load clustering parameters and fit parameters
    clustering_params = load_parameters(params_path)
    fit_params = load_parameters(fit_params_path)

    # Load and preprocess your data (assuming data is loaded as DataFrame)
    data = pd.read_csv(input_csv)

    # Apply scaling if specified
    if scaling_option == 'standard':
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        save_as_pickle(scaler, output_dir + '/scaler.pkl')
    elif scaling_option == 'minmax':
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        save_as_pickle(scaler, output_dir + '/scaler.pkl')
    else:
        data_scaled = data

    # Instantiate clustering model with parameters
    clustering_model = clustering_class(**clustering_params)

    # Fit the clustering model
    clustering_model.fit(data_scaled, **fit_params)

    # Predict clusters
    cluster_labels = clustering_model.predict(data_scaled)

    # Turn data_scaled into pandas dataframe
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    # Save cluster labels and data to CSV
    data_with_clusters = pd.concat([data_scaled, pd.DataFrame({'Cluster': cluster_labels})], axis=1)
    data_with_clusters.to_csv(output_dir + '/data_with_clusters.csv', index=False)

    # Save clustering model
    save_as_pickle(clustering_model, output_dir + '/clustering_model.pkl')

    print("Clustering completed. Clustered data saved to 'data_with_clusters.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform data clustering.')
    parser.add_argument('--class_name', type=str, help='Class name (e.g., sklearn.cluster.KMeans)')
    parser.add_argument('--params_path', type=str, help='Path to clustering parameters JSON file')
    parser.add_argument('--fit_params_path', type=str, help='Path to fit parameters JSON file')
    parser.add_argument('--scaling_option', type=str, help='Scaling option (standard, minmax, none)')
    parser.add_argument('--input_csv', type=str, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving results')
    args = parser.parse_args()

    cluster_data(args.class_name, args.params_path, args.fit_params_path, args.scaling_option, args.input_csv, args.output_dir)
