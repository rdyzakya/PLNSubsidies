import pandas as pd
import argparse
import json
from sklearn.decomposition import PCA

def perform_pca(dataset_path, label_column, pca_params, output_file):
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Extract features (excluding label column)
    features = data.drop(columns=[label_column])

    # Perform PCA
    pca = PCA(n_components=2, **pca_params)
    principal_components = pca.fit_transform(features)

    # Correlation analysis
    correlations = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)
    correlations['PC1_abs'] = abs(correlations['PC1'])
    correlations['PC2_abs'] = abs(correlations['PC2'])
    sorted_features = correlations.sort_values(by=['PC1_abs', 'PC2_abs'], ascending=[False, False])

    # Save sorted feature names to output file
    sorted_features[['PC1', 'PC2']].to_csv(output_file, index=True)
    print(f"PCA and correlation analysis completed. Sorted feature names saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform PCA and correlation analysis on a dataset.')
    parser.add_argument('--dataset_path', type=str, help='Path to the input dataset CSV file')
    parser.add_argument('--label_column', type=str, help='Column name for cluster or label (to exclude from features)')
    parser.add_argument('--pca_params', type=str, help='Path to JSON file containing PCA parameters')
    parser.add_argument('--output_file', type=str, help='Path to save sorted feature names')
    args = parser.parse_args()

    # Load PCA parameters from JSON file
    with open(args.pca_params, 'r') as json_file:
        pca_params = json.load(json_file)

    perform_pca(args.dataset_path, args.label_column, pca_params, args.output_file)
