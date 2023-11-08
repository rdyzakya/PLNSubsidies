import pandas as pd
import argparse
import json
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

distinct_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d',
    '#9edae5', '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939',
    '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52',
    '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173',
    '#a55194', '#ce6dbd', '#de9ed6'
]


def save_scatter(df, out_path):
    # Determine the number of clusters from the unique values in 'Cluster' column

    # Create a list of unique cluster labels
    cluster_labels = df['Cluster'].unique()

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for label in cluster_labels:
        cluster_data = df[df['Cluster'] == label]
        axes[0].scatter(cluster_data['PC1'], cluster_data['PC2'], color=distinct_colors[label], label=f'Cluster {label}')
    
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    ax = fig.add_subplot(122, projection='3d')
    for label in cluster_labels:
        cluster_data = df[df['Cluster'] == label]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'], color=distinct_colors[label], label=f'Cluster {label}')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Set plot titles
    axes[0].set_title('2D Scatter Plot (PC1 vs PC2)')
    ax.set_title('3D Scatter Plot (PC1 vs PC2 vs PC3)')

    plt.legend()

    # Display the plot
    plt.tight_layout()

    # Save the plot to a JPG file
    plt.savefig(out_path)

def perform_pca(dataset_path, label_column, pca_params, output_dir):
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Extract features (excluding label column)
    label_or_cluster = data[label_column]
    features = data.drop(columns=[label_column])

    # Perform PCA
    pca = PCA(n_components=3, **pca_params)
    principal_components = pca.fit_transform(features)

    # Save principal components
    principal_components = pd.DataFrame(principal_components, columns=['PC1', 'PC2', 'PC3'])
    principal_components["Cluster"] = label_or_cluster
    save_scatter(principal_components, os.path.join(output_dir, "scatter.jpg"))


    # Correlation analysis
    correlations = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=features.columns)
    correlations['PC1_abs'] = abs(correlations['PC1'])
    correlations['PC2_abs'] = abs(correlations['PC2'])
    correlations['PC3_abs'] = abs(correlations['PC3'])
    correlations['sum_PC_abs'] = correlations['PC1_abs'] + correlations['PC2_abs'] + correlations['PC3_abs']
    sorted_features = correlations.sort_values(by=['PC1_abs', 'PC2_abs', 'PC3_abs'], ascending=[False, False, False])
    sorted_features.drop(columns=['PC1_abs', 'PC2_abs', 'PC3_abs'], inplace=True)

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
