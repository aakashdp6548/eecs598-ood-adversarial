import torch
from train import main
from argparse import Namespace
import test
from vocab import Vocab
import numpy as np

args = {
    'train': 'data/mnli/train.txt',
    'valid': 'data/mnli/dev.txt',
    'model_type': 'aae',
    'lambda_adv': 10,
    'lambda_p': 0,
    'lambda_kl': 0,
    'noise': [0.3, 0, 0, 0],
    'save_dir': 'checkpoints/aae_epoch24',
    'epochs': 100,
    'load_model': '',
    'vocab_size': 50000,
    'dim_z': 128,
    'dim_emb': 512,
    'dim_h': 1024,
    'nlayers': 1,
    'dim_d': 512,
    'dropout': 0,
    'lr': 0.0005,
    'batch_size': 256,
    'seed': 598,
    'log_interval': 100,
    'no_cuda': False,
}
args = Namespace(**args)

main(args)