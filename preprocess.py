import csv
import argparse

input0 = 'glue_data/MNLI/binary_CN-E/dev.raw.input0'
input1 = 'glue_data/MNLI/binary_CN-E/dev.raw.input1'
labels = 'glue_data/MNLI/binary_CN-E/dev.label'

output = 'glue_data/MNLI/binary_CN-E/dev.tsv'

with open(input0) as fh0, open(input1) as fh1, open(labels) as label_file, open(output, 'wt') as out:
    tsv_writer = csv.writer(out, delimiter='\t')
    for x, y, label in zip(fh0, fh1, label_file):
        label = label.strip()
        if label == 'neutral' or label == 'contradiction':
            label = 'not_entailment'
        
        tsv_writer.writerow([x.strip(), y.strip(), label.strip()])
