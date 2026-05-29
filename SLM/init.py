import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random
import json

import models
import measures

class CharacterLevelTokenizer:
    """
    Tokenize text data at the level of the character.
    
    Args:
        data: The text corpus.
    """
    def __init__(self, data):
        self.data = data
        self.vocab = sorted(list(set(self.data)))
        self.vocab_size = len(self.vocab)

        self.i_to_s = { i:ch for i,ch in enumerate(self.vocab)}
        self.s_to_i = { ch:i for i,ch in self.i_to_s.items()}

    def encode(self,s):
        return [self.s_to_i[c] for c in s]

    def decode(self,s):
        return ''.join([self.i_to_s[i] for i in s])


class MyTextDataLoader:

    def __init__(self, B, T, tokens):

        self.B = B
        self.T = T
        # Always store tokens as torch.long tensor
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.num_batches = (len(self.tokens) - 1) // (B * T)
        print(f"loaded {len(self.tokens)} tokens, split into {self.num_batches} batches")
        self.current_batch = 0

    def __len__(self) -> int:
        # Number of batches available
        return self.num_batches

    def reset(self):
        self.current_batch = 0

    # TODO: build different get function to sample batch at random
    def next_batch(self):

        B, T = self.B, self.T
        start = self.current_batch
        end = self.current_batch + B * T + 1
        # Protect against overflow
        if end > len(self.tokens):
            self.reset()
            start = 0
            end = B * T + 1
        inputs = self.tokens[start:end]
        targets = (inputs[1:]).view(B, T)
        inputs = (inputs[:-1]).view(B, T)
        self.current_batch += B * T

        return inputs, targets  # already long


def init_data(config):
    """
    Initialise dataset.
    
    Returns:
        Tokenizer,
        Two dataloaders for train and validation set.
    """

    if "rhm" in config.dataset:
        with open(config.path+config.tokenizer, 'r') as f:
                tokenizer = json.load(f)
        config.vocab_size = tokenizer['vocab_size'] + 1
        config.eos_token_id = tokenizer['eos_token_id']
        config.unk_token_id = None

    else:
        with open(config.path+config.tokenizer, 'r') as f:
                tokenizer = json.load(f)
        config.vocab_size = tokenizer['vocab_size']
        config.eos_token_id = tokenizer['eos_token_id']
        config.unk_token_id = tokenizer['unk_token_id']

    print("vocabulary size:", config.vocab_size)

    train_corpus = np.load(config.path+config.dataset+".train.npy")
    print("number of training tokens:", len(train_corpus))
    init_index =  random.randint(0,len(train_corpus)-config.train_size)
    train_loader = MyTextDataLoader( config.batch_size, config.block_size, torch.tensor(train_corpus[init_index:init_index+config.train_size+1]))

    valid_corpus = np.load(config.path+config.dataset+".valid.npy")
    print("number of validation tokens:", len(valid_corpus))
    init_index = random.randint(0,len(valid_corpus)-config.val_size)
    val_loader = MyTextDataLoader( config.batch_size, config.block_size, torch.tensor(valid_corpus[init_index:init_index+config.val_size+1]))

    if config.online:
        config.max_steps = train_loader.num_batches
    else:
        assert config.max_epochs is not None, "max_epochs arg required unless training online!"
        config.max_steps = config.max_epochs * train_loader.num_batches
    print(f"Training for {config.max_steps} steps")

    return tokenizer, train_loader, val_loader


def init_model(config, seed=None):
    """
    Initialise machine-learning model.
    Dispatches on config.model: 'gpt2' (default) or 'mamba'.
    """
    seed = config.seed_model if seed is None else seed
    torch.manual_seed(seed)

    if config.model == 'mamba':
        model = models.MambaLM(
            vocab_size=config.vocab_size,
            d_model=config.d_embedding,
            depth=config.depth,
            d_state=getattr(config, 'd_state', 16),
            d_conv=getattr(config, 'd_conv', 4),
            expand=getattr(config, 'mamba_expand', 2),
            dropout=config.dropout,
            share_emb=False,
        )
    else:
        model = models.CLM(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            embedding_dim=config.d_embedding,
            num_heads=config.n_heads,
            ffwd_size=config.ffwd_size,
            num_layers=config.depth,
            dropout=config.dropout,
            rope=config.rope,
            share_emb=False,
        )

    model.to(config.device)

    # model = torch.compile(model)  #TODO: check that this is actually running faster on gpus

    param_count = sum([p.numel() for p in model.parameters()])
    print("# parameters:", param_count)

    return model


def init_training( model, config):
    """
    Initialise training algorithm.
    """
    criterion = nn.CrossEntropyLoss( reduction='mean')
    
    if config.optim =='adam':
        optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.l2
        )
    else:
        raise ValueError("optimizer is invalid (adamW)!")

    if config.scheduler =='cosine':
        scheduler = CosineWarmupLR(
            optimizer, config.warmup_time, config.decay_time, config.decay_factor
        )
    else:
        raise ValueError("scheduler is invalid!")


    return criterion, optimizer, scheduler


def log2ckpt( end, freq):
    """
    Initialise log-spaced iterator.

    Returns:
        List with integer steps spaced multiplicatively by 2**(1/freq) until end.
    """
    current = 1.
    factor = 2**(1./freq)
    threshold = 2**(math.ceil(math.log(1./(factor-1)))+1)
    checkpoints = []

    while current < threshold:
        checkpoints.append( round( current))
        current += 1

    while round(current) < end:
        checkpoints.append( round( current))
        current *= factor

    checkpoints.append( round( end))

    return checkpoints

def init_loglinckpt( step, end, freq):
    """
    Initialise checkpoint iterator.

    Returns:
        Two iterators, one for linear and one for logscale. The iterators coincide upt to some multiple of step, 
        then one proceeds linearly in multiples of step and the other logarithmically in factors of 2**(1/freq).
    """
    # find the correct multiplier
    factor = 2**(1./freq)
    multiplier = 2**(math.ceil(math.log(1./(factor-1)))+1)

    # build log2ckpt lists until multiplier*step
    lin_ckpts = log2ckpt( multiplier*step, freq)
    log_ckpts = lin_ckpts.copy()

    # fill the linear list by adding steps until end
    current = lin_ckpts[-1] + step
    while current <= end:
        lin_ckpts.append(current)
        current += step
    lin_ckpts.append(0)

    # fill the log list by multiplying factors until end
    current = multiplier*factor
    while round(current)*step < end:
        log_ckpts.append( round(current)*step)
        current *= factor

    log_ckpts.append(round( end))
    log_ckpts.append(0)

    return iter(lin_ckpts), iter(log_ckpts)


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_time, decay_time, min_lr_factor):
        self.warmup = warmup_time
        self.decay = decay_time
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.warmup:
            # Linear warmup from 0 to 1
            return step / self.warmup
        elif step < self.decay:
            # Cosine decay to min_lr_factor
            decay_step = step - self.warmup
            total_decay = self.decay - self.warmup
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / total_decay))
            return self.min_lr_factor + (1 - self.min_lr_factor) * cosine_decay
        else:
            # Stay constant after decay
            return self.min_lr_factor
