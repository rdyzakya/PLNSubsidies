import pandas as pd
import argparse
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import pairwise_distances
import pickle
import json

def evaluate_clustering(input_dir, output_file):
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
    
    # Save silhouette score to a file
    with open(output_file, 'w') as fp:
        json.dump(metrics, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustered data.')
    parser.add_argument('--input_dir', type=str, help='Path to clustered data and model directory')
    parser.add_argument('--output_file', type=str, help='Path to save the silhouette score')
    args = parser.parse_args()

    evaluate_clustering(args.input_dir, args.output_file)
