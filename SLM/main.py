import os
import sys
import time
import copy
sys.path.append('~/SLM')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random

import functools
import argparse

import models, init, measures, train


def run( config):

    print(f"Loading {config.path+config.dataset} ... ")
    tokenizer, train_loader, val_loader = init.init_data(config)

    if config.measure_train:
        extrain_loader = copy.deepcopy(train_loader)

    model = init.init_model( config)
    best_model = copy.deepcopy( model).to('cpu')

    criterion, optimizer, scheduler = init.init_training( model, config)

    dynamics = []
    val_loss = measures.evaluate(model, criterion, val_loader, val_loader.num_batches, config.device)
    dynamics.append({'t': 0, 'running': math.log(config.vocab_size), 'val_loss': val_loss})
    if config.measure_train:
        train_loss = measures.evaluate(model, criterion, extrain_loader, extrain_loader.num_batches, config.device)
        dynamics[0]['train_loss'] = train_loss
    if config.loss_by_token:
        ngram_loss = measures.loss_by_token( model, criterion, val_loader, config.block_size, val_loader.num_batches, config.device)
        dynamics[-1]['ngram_loss'] = ngram_loss
    best = {'step':0, 'loss': val_loss, 'model': best_model.state_dict()}

    if config.checkpoints:
        save_model = copy.deepcopy(model).to('cpu')
        torch.save(
            {'model': save_model.state_dict(), 'state': dynamics[-1], 'step':  0},
            f"{config.outname}_t0.pt"
        )

    print_ckpts, save_ckpts = init.init_loglinckpt( config.print_freq, config.max_steps, freq=config.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    print("Valid loss at init: ", val_loss, f"(compare with {math.log(config.vocab_size)})")

    for step in range(config.max_steps):

        t0 = time.time()
        loss = train.train_step( model, train_loader, criterion, optimizer, scheduler, config.device)
        t1 = time.time()
        dt = (t1-t0)*1000

        if (step+1)==print_ckpt:

            val_loss = measures.evaluate(model, criterion, val_loader, val_loader.num_batches, config.device)
            if val_loss<best['loss']:   # update best model if loss is smaller
                best['step'] = step
                best['loss'] = val_loss
                best_model = copy.deepcopy( model).to('cpu')
                best['model'] = best_model.state_dict()

            print(f'step {step+1}, running loss {loss}, dt={dt:.2f}ms, validation loss {val_loss}')
            print_ckpt = next(print_ckpts)

            if (step+1)>=save_ckpt:

                print(f'Evaluating at {step+1} ...')
                dynamics.append({'t': step+1, 'running': loss, 'val_loss': val_loss})

                if config.measure_train:
                    train_loss = measures.evaluate(model, criterion, extrain_loader, extrain_loader.num_batches, config.device)
                    dynamics[-1]['train_loss'] = train_loss
                if config.loss_by_token:
                    ngram_loss = measures.loss_by_token( model, criterion, val_loader, config.block_size, val_loader.num_batches, config.device)
                    dynamics[-1]['ngram_loss'] = ngram_loss
                if config.checkpoints:
                    save_model = copy.deepcopy(model).to('cpu')
                    torch.save(
                        {'model': save_model.state_dict(), 'state': dynamics[-1], 'step':  step+1},
                        f"{config.outname}_t{step+1}.pt"
                    )

                torch.save(
                    {'config': config, 'dynamics': dynamics, 'best':  best},
                    f"{config.outname}.pt"
                )

                save_ckpt = next(save_ckpts)

    if (step+1)<save_ckpt:

        print(f'Evaluating at {step+1} ...')
        dynamics.append({'t': step+1, 'running': loss, 'val_loss': val_loss})

        if config.measure_train:
            train_loss = measures.evaluate(model, criterion, extrain_loader, extrain_loader.num_batches, config.device)
            dynamics[-1]['train_loss'] = train_loss
        if config.loss_by_token:
            ngram_loss = measures.loss_by_token( model, criterion, val_loader, config.block_size, val_loader.num_batches, config.device)
            dynamics[-1]['ngram_loss'] = ngram_loss
        if config.checkpoints:
            save_model = copy.deepcopy(model).to('cpu')
            torch.save(
                {'model': save_model.state_dict(), 'state': dynamics[-1], 'step':  step+1},
                f"{config.outname}_t{step+1}.pt"
            )

        torch.save(
            {'config': config, 'dynamics': dynamics, 'best':  best},
            f"{config.outname}.pt"
        )

    return None


torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Training a Small Language Model on sequence data')
parser.add_argument('--device', type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--path', type=str, default="datasets/")
parser.add_argument('--tokenizer', type=str, default=None)

parser.add_argument('--block_size', metavar='T', type=int, help='sequence length in tokens')
parser.add_argument('--batch_size', metavar='B', type=int)
parser.add_argument('--train_size', metavar='P', type=int)
parser.add_argument('--val_size', metavar='Val', type=int)

'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'mamba'], help='model architecture')
parser.add_argument('--d_embedding', type=int, help='embedding / model dimension')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--seed_model', type=int, help='seed for model initialization')
# transformer-specific
parser.add_argument('--n_heads', type=int, help='number of attention heads (gpt2 only)')
parser.add_argument('--ffwd_size', type=int, help='MLP width multiplier (gpt2 only)')
parser.add_argument('--rope', default=False, action='store_true')
# mamba-specific
parser.add_argument('--d_state', type=int, default=16, help='SSM state dimension (mamba only)')
parser.add_argument('--d_conv', type=int, default=4, help='depthwise conv kernel size (mamba only)')
parser.add_argument('--mamba_expand', type=int, default=2, help='inner-dim expansion factor (mamba only)')
'''
	TRAINING ARGS
'''
parser.add_argument('--online', default=False, action='store_true')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--scheduler', type=str, default='cosine')
parser.add_argument('--warmup_time', type=int, default=16)
parser.add_argument('--decay_time', type=int, default=128)
parser.add_argument('--decay_factor', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--l2', type=float, help='l2 regularisation', default=0.0)
parser.add_argument('--max_epochs', type=int, default=1)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=16)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=2)
parser.add_argument('--measure_train', default=False, action='store_true')
parser.add_argument('--loss_by_token', default=False, action='store_true')
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

config = parser.parse_args()
run( config)
