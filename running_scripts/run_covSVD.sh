#!/bin/bash

vocab_size=8192
split=train

max_dist=$1
dist=$2

plist='32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456'

for num_data in ${plist};
do
    python3 covarianceSVD.py --device cpu --path ../../results/tinystories/correlations/ --filename correlations_tinystories.BPE${vocab_size}.${split}_d${max_dist}_t${dist} --num_data ${num_data} --outname CovSVD_tinystories.BPE${vocab_size}.${split}_d${max_dist}_t${dist}
done