import os
import importlib
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def save_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def cluster_data(config_path, scaling_option, input_csv, output_dir):

    config = load_json(config_path)
    # Dynamically import clustering class
    module_name, class_name = config["class_name"].rsplit('.', 1)
    module = importlib.import_module(module_name)
    clustering_class = getattr(module, class_name)

    # Load clustering parameters and fit parameters
    clustering_params = config["hparams"]
    fit_params = config["fit_params"]

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
    parser.add_argument('--config_path', type=str, help='Path to config JSON file, containing class_name, hparams, and fit_params keys')
    parser.add_argument('--scaling_option', type=str, help='Scaling option (standard, minmax, none)')
    parser.add_argument('--input_csv', type=str, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving results')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cluster_data(args.config_path, args.scaling_option, args.input_csv, args.output_dir)
