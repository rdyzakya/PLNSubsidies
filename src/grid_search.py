import os
import subprocess
from itertools import product
from copy import deepcopy
from utils import load_json, dump_json

def run_process(command):
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
    print("Output of eval.py:")
    print(output)
    return output

gs_path = "../config/gs.json"
config_path = "../config/config.json"
data_path = "../data/preprocessed_data_all.csv"
out_dir = "./out"

gs_config = load_json(gs_path)

max_sil_score = -1
best_candidate = {}

for gsc in gs_config:

    class_name = gsc["class_name"]
    hparams = gsc["hparams"]
    fit_params = gsc["fit_params"]

    for k, v in hparams.items():
        if not isinstance(v, list):
            hparams[k] = [v]
    product_results = list(product(*hparams.values()))
    candidate = [{key: value for key, value in zip(hparams.keys(), product)} for product in product_results]

    for c in candidate:
        config = {
            "class_name" : class_name,
            "hparams" : c,
            "fit_params" : fit_params
        }
        dump_json(config, config_path)

        for scaling in ["standard", "minmax", "none"]:
            
            out_foldername = [class_name]
            for k, v in c.items():
                out_foldername.append(f"{k}={v}")
            out_foldername.append(f"scaling={scaling}")
            out_foldername = '_'.join(out_foldername)
            out_path = os.path.join(out_dir, out_foldername)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            current_candidate = deepcopy(config)
            current_candidate["scaling"] = scaling
            dump_json(current_candidate, os.path.join(out_path, "config.json"))

            try:
                # TRAIN
                # python ./src/train.py --config_path ./config/example.json --scaling_option standard --input_csv ./example.csv --output_dir ./out
                print(f"Clustering for {class_name} with hparams {str(c)} and scaling {scaling}...")
                train_command = ["python", "train.py", "--config_path", config_path, "--scaling_option", scaling, "--input_csv", data_path, "--output_dir", out_path]
                train_output = run_process(train_command)
                
                # EVAL
                # python ./src/eval.py --input_csv ./out/data_with_clusters.csv --output_file ./out/result.txt
                print("Evaluation...")
                sil_score_path = os.path.join(out_path, "silhouette_score.txt")
                eval_command = ["python", "eval.py", "--input_csv", os.path.join(out_path, "data_with_clusters.csv"), "--output_file", sil_score_path]
                eval_output = run_process(eval_command)

                ## get score
                with open(sil_score_path, 'r') as fp:
                    sil_score = float(fp.read().strip())
                if sil_score > max_sil_score:
                    max_sil_score = sil_score
                    best_candidate = current_candidate

                # PCA
                # python ./src/pca.py --dataset_path ./out/data_with_clusters.csv --output_dir ./out
                print("Performing PCA...")
                pca_command = ["python", "pca.py", "--dataset_path", os.path.join(out_path, "data_with_clusters.csv"), "--output_dir", out_path]
                pca_output = run_process(pca_command)
            except subprocess.CalledProcessError as e:
                print("Error:", e.output)
            print("="*128)

print("Saving best candidate...")
dump_json(best_candidate, os.path.join(out_dir, "best_candidate.json"))