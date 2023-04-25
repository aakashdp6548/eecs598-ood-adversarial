import torch

import test
from vocab import Vocab
import numpy as np

from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import TensorDataset, DataLoader
from transformers import glue_convert_examples_to_features
from transformers import glue_processors
from typing import List, Optional, Union
from dataclasses import dataclass

import pandas as pd
import Levenshtein
import string
import jiwer


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: str
    label: Optional[str] = None
        
@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None


def get_model(path, vocab):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = test.AAE(vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

def encode(sents, vocab, batch_size, model, device, enc='mu'):
    batches, order = test.get_batches(sents, vocab, batch_size, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        if enc == 'mu':
            zi = mu
        else:
            zi = test.reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_

def decode(z, vocab, batch_size, max_len, model, device, dec='sample'):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+batch_size], device=device)
        outputs = model.generate(zi, max_len, dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += batch_size
    return test.strip_eos(sents)

def load_data(premise, hypotheses, tokenizer):
    processor = glue_processors['mnli']()
    label_list = ["contradiction", "entailment", "neutral"]
    examples = []
    for i, hypothesis in enumerate(hypotheses):
        examples.append(InputExample(guid=f'test-{i}', text_a=premise, text_b=hypothesis, label='contradiction'))
    
    label_map = {label: i for i, label in enumerate(label_list)}
    labels = [label_map[example.label] for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=128,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

vocab = Vocab('../checkpoints/aae_epoch100/vocab.txt')
test.set_seed(598)
torch.manual_seed(598)
device = torch.device('cuda')

model = get_model('../checkpoints/aae_epoch100/model.pt', vocab)

perturb_noise = 0.25

classifier_path = '../checkpoints/mnli_baseline_distilbert-2023-04-07_10-48-14/checkpoint-last'
config = DistilBertConfig.from_pretrained(
    classifier_path,
    num_labels=3,
    finetuning_task='mnli',
    attention_probs_dropout_prob=0,
    hidden_dropout_prob=0.1
)
tokenizer = DistilBertTokenizer.from_pretrained(
    classifier_path,
    do_lower_case=True,
)
classifier = DistilBertForSequenceClassification.from_pretrained(
    classifier_path,
    config=config,
    ignore_mismatched_sizes=True
)


data = pd.read_csv('data/mnli/train2.tsv', sep='\t')

aug_batch = []
aug_i = 0
data_start = 0
data_appended = data_start

premise_seen = {}

for index, row in data.loc[data_start:].iterrows():
    # Load premise, hypothesis, and label
    premise_text = row['sentence1']
    raw_premise = row['sentence1_binary_parse'].split(' ')
    try:
        raw_hypothesis = row['sentence2_binary_parse'].split(' ')
    except AttributeError:
        continue
    orig_label = row['gold_label']
        
    # Process premise
    premise_words = []
    for word in raw_premise:
        if word != "(" and word != ")":
            premise_words.append(word)
    premise = " ".join(premise_words)
    
    # Check that premise is unique
    if premise in premise_seen.keys():
        continue
        
    premise_seen[premise] = True

    # Process hypothesis
    hypothesis_words = []
    for word in raw_hypothesis:
        if word != "(" and word != ")":
            hypothesis_words.append(word)
    hypothesis = " ".join(hypothesis_words)
    
    # Generate sentences
    sents = [ hypothesis.split() ]
    z = encode(sents, vocab, 1, model, device)
    n = 10
    
    orig_hypothesis = hypothesis
    hypotheses = []
    for i in range(n):
        z_noise = z + np.random.normal(0, perturb_noise, size=z.shape).astype('f')
        decoded = decode(z_noise, vocab, 1, 30, model, device, dec='greedy')
        hypotheses.append(' '.join(decoded[0]))
        
    # Run classifier on new hypotheses
    dataset = load_data(premise, hypotheses, tokenizer)
    eval_dataloader = DataLoader(dataset, batch_size=16)
    for batch in eval_dataloader:
        classifier.eval()
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        _, logits = classifier(**inputs)[:2]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)

    label_list = ["contradiction", "entailment", "neutral"]
    
    min_dist = 1e9
    best_sent = None
    best_label = None

    for sentence, pred in zip(hypotheses, preds):
        if '<unk>' in sentence:
            continue
        
        # Remove trailing punctuation for comparison
        if sentence[-1] in string.punctuation:
            sentence_comp = sentence[:-1].rstrip()
        else:
            sentence_comp = sentence
        if orig_hypothesis[-1] in string.punctuation:
            orig_hypothesis_comp = orig_hypothesis[:-1].rstrip()
        else:
            orig_hypothesis_comp = orig_hypothesis

        # Choose best sentence based on WER to original (with different label)
        if orig_label != label_list[pred] and orig_hypothesis_comp != sentence_comp:
            dist = jiwer.wer(orig_hypothesis_comp, sentence_comp)
            dist *= len(orig_hypothesis_comp.split())
            if dist <= 2 and dist > 0:
                if dist < min_dist:
                    min_dist = dist
                    best_sent = sentence
                    best_label = label_list[pred]
            
                
    # Skip if no close sentences were found
    if best_sent == None:
        continue

    print('Premise: {}'.format(premise))
    print('Original hypothesis: {} --> {}'.format(orig_hypothesis, orig_label))
    print('Best sentence: {} --> {}\n'.format(best_sent, best_label))

    # Fill aug_data row with necessary info
    aug_row = []
    for i in range(8):
        aug_row.append('')
    aug_row.append(premise_text)
    aug_row.append(best_sent)
    aug_row.append('')
    aug_row.append(orig_label)
    aug_row.append('')

    aug_batch.append(aug_row)
    aug_i += 1
    data_appended += 1
    
    # Write to TSV every 1000 lines in case it's too slow overall
    if aug_i >= 500:
        aug_data = pd.DataFrame(columns=data.columns)

        # Add data from batch
        for batch_i, batch_row in enumerate(aug_batch):
            aug_data.loc[batch_i] = batch_row
        
        aug_data.to_csv('data/aug{}.tsv'.format(data_appended), sep="\t")
        
        # Reset aug data and counter
        aug_data = pd.DataFrame(columns=data.columns)
        aug_batch = []
        aug_i = 0
        
    if data_appended >= data.shape[0] * 0.05:
        break
