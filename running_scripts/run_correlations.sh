#!/bin/bash

vocab_size=8192

max_dist=$1
dist=$2
num_data=$3
batch_size=$4

python3 correlations.py --device cpu --vocab_size ${vocab_size} --path ../datasets/tinystories/ --filename tinystories.BPE8192.train.npy --max_dist ${max_dist} --dist ${dist} --num_data ${num_data} --batch_size ${batch_size} --outname correlations_tinystories.BPE${vocab_size}.train_d${max_dist}_t${dist}