import pandas as pd
import argparse
import json
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def save_scatter(df, out_path):
    # Determine the number of clusters from the unique values in 'Cluster' column
    num_clusters = df['Cluster'].nunique()

    # Create a list of unique cluster labels
    cluster_labels = df['Cluster'].unique()

    # Plotting the clusters dynamically based on the number of clusters
    plt.figure(figsize=(8, 6))

    # Iterate through each cluster and plot the data points
    for label in cluster_labels:
        cluster_data = df[df['Cluster'] == label]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {label}')

    # Set labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Cluster Visualization')

    # Add legend
    plt.legend()

    # Save the plot to a JPG file
    plt.savefig(out_path)

def perform_pca(dataset_path, label_column, pca_params, output_dir):
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Extract features (excluding label column)
    label_or_cluster = data[label_column]
    features = data.drop(columns=[label_column])

    # Perform PCA
    pca = PCA(n_components=2, **pca_params)
    principal_components = pca.fit_transform(features)

    # Save principal components
    principal_components = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    principal_components["Cluster"] = label_or_cluster
    save_scatter(principal_components, os.path.join(output_dir, "scatter.jpg"))


    # Correlation analysis
    correlations = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)
    correlations['PC1_abs'] = abs(correlations['PC1'])
    correlations['PC2_abs'] = abs(correlations['PC2'])
    correlations['sum_PC_abs'] = correlations['PC1_abs'] + correlations['PC2_abs']
    sorted_features = correlations.sort_values(by=['PC1_abs', 'PC2_abs'], ascending=[False, False])
    sorted_features.drop(columns=['PC1_abs', 'PC2_abs'], inplace=True)

    # Save sorted feature names to output file
    corr_path = os.path.join(output_dir,"corr.csv")
    sorted_features.to_csv(corr_path, index=True)
    print(f"PCA and correlation analysis completed. Sorted feature names saved to '{corr_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform PCA and correlation analysis on a dataset.')
    parser.add_argument('--dataset_path', type=str, help='Path to the input dataset CSV file')
    parser.add_argument('--label_column', type=str, help='Column name for cluster or label (to exclude from features)', default="Cluster")
    parser.add_argument('--pca_params', type=str, help='Path to JSON file containing PCA parameters')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    args = parser.parse_args()

    # Load PCA parameters from JSON file
    pca_params = {}
    if args.pca_params:
        with open(args.pca_params, 'r') as json_file:
            pca_params = json.load(json_file)

    perform_pca(args.dataset_path, args.label_column, pca_params, args.output_dir)
