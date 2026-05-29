import numpy as np
import math
import random

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

def run( args):

    stats = torch.load(args.path+args.filename+f"_P{args.num_data}.pt")

    A = stats['joint'] - stats['right'][:, None] @ stats['left'][None]
    A = A.numpy()
    U, S, Vt = np.linalg.svd(A)

    torch.save(
        {'P': args.num_data, 'U': torch.from_numpy(U).to(torch.float32), 'S': torch.from_numpy(S).to(torch.float32), 'Vt': torch.from_numpy(Vt).to(torch.float32)},
        args.path+args.outname+f"_P{args.num_data}.pt"
    )
    svals_sorted = -np.sort(-S)
    print(args.filename, svals_sorted[0], (svals_sorted**2).sum()**.5)

    return None


torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Compute covariance SVD from joint and marginal statistics')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--path', type=str, required=True, help='path of the dataset')
parser.add_argument('--filename', type=str, required=True, help='filename of the dataset')
parser.add_argument('--num_data', type=int, required=True, help='number of data')
'''
	OUTPUT ARGS
'''
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

args = parser.parse_args()
run( args)