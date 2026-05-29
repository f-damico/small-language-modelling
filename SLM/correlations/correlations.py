import numpy as np
import math
import random

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

def run( args):

    vocab_size = args.vocab_size
    corpus = np.load(args.path+args.filename)
    corpus = torch.tensor(corpus).to(args.device)
    corpus_len = corpus.size(0)

    max_dist = args.max_dist
    dist = args.dist
    assert dist < max_dist, f'dist must be smaller than max_dist={max_dist}!'

    print('total dataset size: ', args.num_data, f'(corpus length = {corpus_len})')
    if args.batch_size is None:
        args.batch_size = args.num_data
    num_batches = args.num_data//args.batch_size
    samples = torch.tensor( random.sample( range(max_dist-1,corpus_len), args.num_data))

    joint = torch.zeros( # right_feature x left_feature
        (vocab_size, vocab_size), device=args.device, dtype=torch.int64
    )
    right = torch.zeros(
    (vocab_size), device=args.device, dtype=torch.int64
    )
    left = torch.zeros(
    (vocab_size), device=args.device, dtype=torch.int64
    )
    ckpt = 1

    for i in range(num_batches):

        samples_batch = samples[args.batch_size*i:args.batch_size*(i+1)]
        sample1 = corpus[samples_batch].long()
        sample2 = corpus[samples_batch-dist].long()

        right += torch.bincount(sample1, minlength=vocab_size)
        left += torch.bincount(sample2, minlength=vocab_size)

        pair_ids = sample1*vocab_size + sample2
        # NOTE: the reshape is row-first
        joint += torch.bincount(pair_ids, minlength=vocab_size*vocab_size).reshape(vocab_size, vocab_size)

        if (i+1)==ckpt:

            P = ckpt*args.batch_size
            prob_right = right.to(torch.float32)/P
            prob_left = left.to(torch.float32)/P
            prob_joint = joint.to(torch.float32)/P
            corr = (prob_joint-right[:,None]@left[None,:]).std()

            print(f'saving at {P} data')
            torch.save(
                {'P': P, 'right': prob_right, 'left': prob_left, 'joint': prob_joint, 'corr': corr},
                f"{args.outname}_P{ckpt*args.batch_size}.pt"
            )
            ckpt *= 2

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Compute two-point correlations from tokenized dataset')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--vocab_size', metavar='v', type=int, required=True, help='vocabulary size')
parser.add_argument('--path', type=str, required=True, help='path of the dataset')
parser.add_argument('--filename', type=str, required=True, help='filename of the dataset')
parser.add_argument('--max_dist', type=int, required=True, help='maximal token distance')
parser.add_argument('--dist', metavar='t', type=int, required=True, help='token distance')
parser.add_argument('--num_data', type=int, required=True, help='number of data')
parser.add_argument('--batch_size', type=int, default=128)
'''
	OUTPUT ARGS
'''
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

args = parser.parse_args()
run( args)