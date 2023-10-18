python ./src/train.py --config_path ./config/example.json --scaling_option standard --input_csv ./data/archive/data.csv --output_dir ./out
python ./src/eval.py --input_csv ./out/data_with_clusters.csv --output_file ./out/result.txt
python ./src/pca.py --dataset_path ./out/data_with_clusters.csv --output_dir ./out