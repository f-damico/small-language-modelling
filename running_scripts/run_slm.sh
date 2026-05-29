#!/bin/bash

dataset=tinystories
version=BPE8192

block_size=128
batch_size=128
train_size=134217728
val_size=524288

d_embedding=$1
n_heads=$2
ffwd_size=4
depth=$3
seed=$(od -An -N3 -i /dev/random | tr -d '[:space:]')
echo ${seed}

optim=adam
lr=3e-4
warmup_time=8
decay_time=$5 # usually two epochs
max_epochs=$6

print=1024
save=2

grun python3 SLM/main.py --device cuda --dataset ${dataset}.${version} --path SLM/datasets/${dataset}/ --tokenizer ${dataset}.${version}.meta.json --block_size ${block_size} --batch_size ${batch_size} --train_size ${train_size} --val_size ${val_size} --d_embedding ${d_embedding} --n_heads ${n_heads} --ffwd_size ${ffwd_size} --depth ${depth} --seed_model ${seed} --lr ${lr} --scheduler cosine --warmup_time ${warmup_time} --decay_time ${decay_time} --max_epochs ${max_epochs} --print_freq ${print} --save_freq ${save} --loss_by_token --outname tinystories-BPE8192_T${block_size}_P${train_size}_d${depth}_demb${d_embedding}_nh${n_heads}_bs${batch_size}_lr${lr}_wt${warmup_time}_dt${decay_time}_${seed}
