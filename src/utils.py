import json
import pickle

def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def dump_json(obj, json_path):
    with open(json_path, 'w') as file:
        json.dump(obj, file)

def save_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)