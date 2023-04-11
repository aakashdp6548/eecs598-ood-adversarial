import csv
import argparse
import os
import json
import codecs

def two_label_MNLI():
    '''
    Converts MNLI data to 2-label format, combining contradiction and neutral to 'not_entailment'.
    '''
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

def preprocess_snli(in_path, write_path):
    '''
    Writes SNLI data in the same format as MNLI to use with MNLIProcessor.
    '''
    with codecs.open(in_path, encoding='utf-8') as f:
        lines = f.readlines()

    headings = [
        'index',
        'promptID',
        'pairID',
        'genre',
        'sentence1_binary_parse',
        'sentence2_binary_parse',
        'sentence1_parse',
        'sentence2_parse',
        'sentence1',
        'sentence2',
        'label1','label2','label3','label4','label5',
        'gold_label']

    with open(write_path, 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(headings)

        index = 0
        for line in lines:
            loaded_example = json.loads(line)
            label = loaded_example['gold_label'].strip()
            if label == '-':    # skip '-' labels
                continue
            premise = loaded_example['sentence1']
            hypothesis = loaded_example['sentence2']
            tsv_writer.writerow([index, '', '', '', '', '', '', '', premise, hypothesis, '', '', '', '', '', label])
            index += 1

    print(f'Finished writing to {write_path}')

def mnli_for_gan(train_path, dev_path, write_path):
    '''
    Writes MNLI data in appropriate format for baseline models for GAN
    used in GAN_text/train_baseline.py.
    '''
    sent_filename = os.path.join(write_path, 'sentences.dlnlp')
    train_filename = os.path.join(write_path, 'train.txt')
    test_filename = os.path.join(write_path, 'test.txt')
    
    with open(sent_filename, 'w') as sent_file:
        with open(train_path, 'r') as in_file, open(train_filename, 'w') as train_file:
            index = 0
            for line in in_file:
                line = line.strip().split('\t')
                premise = line[8]
                hypothesis = line[9]
                label = line[-1]

                sent_file.write(f'{index}\t{premise}\n')
                sent_file.write(f'{index + 1}\t{hypothesis}\n')
                train_file.write(f'{label}\t{index}\t{index + 1}\n')
                index += 2

        with open(dev_path, 'r') as in_file, open(test_filename, 'w') as test_file:
            for line in in_file:
                line = line.strip().split('\t')
                premise = line[8]
                hypothesis = line[9]
                label = line[-1]

                sent_file.write(f'{index}\t{premise}\n')
                sent_file.write(f'{index + 1}\t{hypothesis}\n')
                test_file.write(f'{label}\t{index}\t{index + 1}\n')
                index += 2
