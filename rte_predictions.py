import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pred', type=str, required=True, help='Path to file with predictions and labels')
args = parser.parse_args()

# open file and read pred and label lines
if not os.path.isfile(args.pred):
    raise Exception('Cannot find {args.pred}')

with open(args.pred) as f:
    lines = f.readlines()

pred = lines[0].strip().replace(',', ' ').split()
pred = np.asarray(pred, dtype=float)
labels = lines[1].strip().replace(',', ' ').split()
labels = np.asarray(labels, dtype=float)

# Convert MNLI-based predictions to RTE labels
# MNLI: 0 = contradiction, 1 = entailment, 2 = neutral
# RTE: 0 = entail, 1 = not entail
pred = np.abs(pred - 1)  # 1 -> 0, 0 and 2 -> 1

acc = np.sum(pred == labels) / pred.shape[0]
print(f'Accuracy: {acc}')