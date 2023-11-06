import os
import json
import pandas as pd
import argparse
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
            "Name" : d,
            "Overall" : utils.overall_metrics(entry)
        })
        result.append(entry)
    result = pd.DataFrame(result)
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
