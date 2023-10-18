import pandas as pd
from sklearn.metrics import silhouette_score
import argparse

def evaluate_clustering(input_csv, output_file):
    # Load clustered data
    clustered_data = pd.read_csv(input_csv)

    # Extract cluster labels and features
    cluster_labels = clustered_data['Cluster']
    features = clustered_data.drop(columns=['Cluster'])

    # Calculate silhouette score
    silhouette_avg = silhouette_score(features, cluster_labels)

    print(f'Silhouette Score: {silhouette_avg}')

    # Save silhouette score to a file
    with open(output_file, 'w') as file:
        file.write(str(silhouette_avg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustered data.')
    parser.add_argument('--input_csv', type=str, help='Path to clustered data CSV file')
    parser.add_argument('--output_file', type=str, help='Path to save the silhouette score')
    args = parser.parse_args()

    evaluate_clustering(args.input_csv, args.output_file)
