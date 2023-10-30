import os
import json
import pickle
import argparse
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

def evaluate_clustering(input_dir, output_dir):
    # Load clustered data
    clustered_data = pd.read_csv(os.path.join(input_dir, "data_with_clusters.csv"))

    # Extract cluster labels and features
    cluster_labels = clustered_data['Cluster']
    features = clustered_data.drop(columns=['Cluster'])
    
    # Load model
    with open(os.path.join(input_dir, "clustering_model.pkl")) as fp:
        model = pickle.load(fp)
    
    # Silhouette Score (higher is better)
    silhouette = silhouette_score(features, cluster_labels)
    
    # Davies-Bouldin Index (lower is better)
    db_index = davies_bouldin_score(features, cluster_labels)
    
    # Dunn Index (higher is better)
    pairwise_distances_matrix = pairwise_distances(features)
    min_intra_cluster_distances = [min(pairwise_distances_matrix[i][cluster_labels == label])
                                   for i, label in enumerate(cluster_labels)]
    max_inter_cluster_distances = [max(pairwise_distances_matrix[i][cluster_labels != label])
                                   for i, label in enumerate(cluster_labels)]
    dunn_index = min(min_intra_cluster_distances) / max(max_inter_cluster_distances)
    
    metrics = {
        'Silhouette Score': silhouette,
        'Davies-Bouldin Index': db_index,
        'Dunn Index': dunn_index,
    }

    if "inertia_" in dir(model):
        metrics["Inertia"] = model.inertia_
    
    print(metrics)
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as fp:
        json.dump(metrics, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustered data.')
    parser.add_argument('--input_dir', type=str, help='Path to clustered data and model directory')
    parser.add_argument('--output_dir', type=str, help='Path to output dir')
    args = parser.parse_args()

    evaluate_clustering(args.input_dir, args.output_dir)
