import os
import shutil
import csv
from flask import Flask, render_template
import argparse

app = Flask(__name__)

# # Load data from CSV
# data = []
# with open('../../out/out_450/summary_score.csv', 'r') as csvfile:
#     csvreader = csv.DictReader(csvfile)
#     for row in csvreader:
#         data.append(row)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score/<score_type>')
def score(score_type):
    score_names = [(row["Name"], float(row[score_type])) for row in data]
    score_names = sorted(score_names, key=lambda x: x[1])
    scores = [el[1] for el in score_names]
    names = [el[0] for el in score_names]
    image_url = f"/static/images/{score_names[0][0]}/scatter.jpg"

    return render_template('score.html', score_type=score_type, image_url=image_url, scores=scores, names=names, length=len(score_names))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization evaluation')
    parser.add_argument('--dir', type=str, help='Path to directory consist of all models created', default='../../out/out_450')
    args = parser.parse_args()

    data = []

    if not os.path.exists("./static"):
        os.makedirs("./static")
    else:
        shutil.rmtree("./static")
        os.makedirs("./static")

    shutil.copy(os.path.join(args.dir, "summary_score.csv"), "./static/summary_score.csv")

    with open('./static/summary_score.csv', 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)
    
    names = [el for el in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, el))]

    for n in names:
        folder = os.path.join("./static/images/", n)
        if not os.path.exists(folder):
            os.makedirs(folder)
        shutil.copy(os.path.join(args.dir, n, "scatter.jpg"), os.path.join(folder, "scatter.jpg"))

    app.run(debug=True)
