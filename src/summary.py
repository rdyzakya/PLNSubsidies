import os
import json
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import utils

def summarize(directory):
    result = []
    listdir = os.listdir(directory)
    for d in listdir:
        if not os.path.isdir(os.path.join(directory, d)):
            continue
        try:
            with open(os.path.join(directory, d, "metrics.json"), 'r') as fp:
                entry = json.load(fp)
        except:
            continue
        entry.update({
            "Name" : d
        })
        result.append(entry)
    result = pd.DataFrame(result)
    result["ch_scaled"] = MinMaxScaler().fit_transform(result["Calinski-Harabasz Index"].values.reshape(1,-1)).reshape(-1,1)
    result["overall"] = summary.apply(overall_metrics, axis=1)
    columns = list(result.columns)
    columns.remove("Name")
    result = result[["Name"] + columns]
    result.to_csv(os.path.join(directory, "summary.csv"), index=False)

    print("Done.., saved in", os.path.join(directory, "summary.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize modeling results.')
    parser.add_argument('--dir', type=str, help='Directory of results.')
    args = parser.parse_args()

    summarize(args.dir)
