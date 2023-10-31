import json
import pickle
import numpy as np

def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def dump_json(obj, json_path):
    with open(json_path, 'w') as file:
        json.dump(obj, file)

def load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def save_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, 1e-12, np.inf)))

def overall_metrics(metrics):
    silhouette = metrics["Silhouette Score"]
    db_index = metrics["Davies-Bouldin Index"]
    ch_index = metrics["Calinski-Harabasz Index"]

    # greater is better
    scaled_silhouette = (silhouette + 1)/2 # -1...1
    # lower is better
    scaled_db_index = -2 * sigmoid(np.abs(db_index)) + 2 # -inf...inf
    # greater is better
    scaled_ch_index = 2 * sigmoid(np.abs(ch_index)) - 1 # 0...inf

    # average
    return (scaled_silhouette + scaled_db_index + scaled_ch_index) / 3