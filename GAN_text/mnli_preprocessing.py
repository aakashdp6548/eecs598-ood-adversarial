import os
import json
import codecs
import argparse
import pandas as pd

"""
Transforms SNLI data into lines of text files
    (data format required for ARAE model).
Gets rid of repeated premise sentences.
"""


def transform_data(in_path):
    print("Loading", in_path)

    premises = []
    hypotheses = []

    last_premise = None

    data = pd.read_csv(in_path, sep='\t')

    for index, row in data.iterrows():
        # load premise and hypothesis
        raw_premise = row['sentence1_binary_parse'].split(' ')
        try:
            raw_hypothesis = row['sentence2_binary_parse'].split(' ')
        except AttributeError:
            continue

        # process premise
        premise_words = []
        # loop through words of premise binary parse
        for word in raw_premise:
            # don't add parse brackets
            if word != "(" and word != ")":
                premise_words.append(word)
        premise = " ".join(premise_words)

        # process hypothesis
        hypothesis_words = []
        for word in raw_hypothesis:
            if word != "(" and word != ")":
                hypothesis_words.append(word)
        hypothesis = " ".join(hypothesis_words)

        # make sure to not repeat premiess
        if premise != last_premise:
            premises.append(premise)
        hypotheses.append(hypothesis)

        last_premise = premise

    return premises, hypotheses


def write_sentences(write_path, premises, hypotheses, append=False):
    print("Writing to {}\n".format(write_path))
    if append:
        with open(write_path, "a") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')
    else:
        with open(write_path, "w") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="datasets/MNLI",
                        help='path to mnli data')
    parser.add_argument('--out_path', type=str, default="datasets/MNLI/processed",
                        help='path to write mnli language modeling data to')
    args = parser.parse_args()

    # make out-path directory if it doesn't exist
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print("Creating directory "+args.out_path)

    # process and write test.txt and train.txt files
    premises, hypotheses = \
        transform_data(os.path.join(args.in_path, "test_matched.tsv"))
    write_sentences(write_path=os.path.join(args.out_path, "test.txt"),
                    premises=premises, hypotheses=hypotheses)

    premises, hypotheses = \
        transform_data(os.path.join(args.in_path, "train2.tsv"))
    write_sentences(write_path=os.path.join(args.out_path, "train.txt"),
                    premises=premises, hypotheses=hypotheses)

    premises, hypotheses = \
        transform_data(os.path.join(args.in_path, "dev_matched.tsv"))
    write_sentences(write_path=os.path.join(args.out_path, "train.txt"),
                    premises=premises, hypotheses=hypotheses, append=True)
