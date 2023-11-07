import os
import json
import argparse
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from utils import load_pickle

def evaluate_clustering(input_dir, output_dir):
    # Load clustered data
    clustered_data = pd.read_csv(os.path.join(input_dir, "data_with_clusters.csv"))

    # Extract cluster labels and features
    cluster_labels = clustered_data['Cluster']
    features = clustered_data.drop(columns=['Cluster'])
    
    # Load model
    model = load_pickle(os.path.join(input_dir, "clustering_model.pkl"))

    try:
    
        # Silhouette Score (higher is better)
        silhouette = silhouette_score(features, cluster_labels)
        
        # Davies-Bouldin Index (lower is better)
        db_index = davies_bouldin_score(features, cluster_labels)

        # Calinski-Harabasz Index 
        ch_index = calinski_harabasz_score(features, cluster_labels)
        
        metrics = {
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': db_index,
            'Calinski-Harabasz Index': ch_index
        }

        if "inertia_" in dir(model):
            metrics["Inertia"] = model.inertia_
    except:
        metrics = {
            'Silhouette Score': -1,
            'Davies-Bouldin Index': 1000,
            'Calinski-Harabasz Index': 0
        }
        
    print(metrics)
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as fp:
        json.dump(metrics, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustered data.')
    parser.add_argument('--input_dir', type=str, help='Path to clustered data and model directory')
    parser.add_argument('--output_dir', type=str, help='Path to output dir')
    args = parser.parse_args()

    evaluate_clustering(args.input_dir, args.output_dir)
