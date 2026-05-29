import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEncoding(nn.Module):
    """
    Rotary Positional Encoding.
    
    Args:
        head_dim: The embedding dimension per head.
        
    """
    def __init__(
        self, head_dim: int, base: float = 10000.0, offset: int = 0
    ):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head dimension"

        self.dim = head_dim
        self.base = base
        self.offset = offset

        # theta (head_dim/2)
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta, persistent=False)

    def _rotate_half(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        # x: (..., head_sim) where head_dim is even
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def _get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions (T)
        pos = torch.arange(self.offset, self.offset + seq_len, device=device, dtype=dtype)
        pos_theta = pos[:,None] @ self.theta[None]
        # pos_theta -> (T, head_dim) by duplicating each freq for the pair
        pos_theta = torch.repeat_interleave(pos_theta, repeats=2, dim=-1)
        pos_cos = pos_theta.cos().to(dtype=dtype)[None, None, :, :]
        pos_sin = pos_theta.sin().to(dtype=dtype)[None, None, :, :]
        return pos_cos, pos_sin

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: The query, tensor of size (batch_size, num_heads, seq_len, head_dim).
            k: The key, tensor of size (batch_size, num_heads, seq_len, head_dim).

        Returns:
            rotated q and k matrices,
            of size (batch_size, num_heads, seq_len, head_dim)
        """
        B,H,T,C = q.size()
        # compute sines and cosines
        cos, sin = self._get_cos_sin(T,device=q.device,dtype=q.dtype)
        q_r = (q * cos) + (self._rotate_half(q) * sin)
        k_r = (k * cos) + (self._rotate_half(k) * sin)

        return q_r, k_r


class AttentionHead(nn.Module):
    """
    One Head of Self-Attention.
    
    Args:
        input_dim: The dimension of input tokens.
        input_size: The (maximal) number of input tokens.
        qk_dim: The dimension of query & key sequences.
        out_dim: The dimension of output tokens.
        dropout: The fraction of weights to zero via dropout.
        decoder: True for one-directional attention.
        
    """
    def __init__(
        self, input_dim, input_size, qk_dim, out_dim, dropout, decoder=True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_size = input_size
        self.out_dim = out_dim
        self.dropout = dropout
        self.decoder = decoder

        self.key=nn.Parameter(
            torch.randn( qk_dim, self.input_dim)
        )
        self.query=nn.Parameter(
            torch.randn( qk_dim, self.input_dim)
        )
        self.value=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        if decoder:
            self.register_buffer('tril', torch.tril(torch.ones(self.input_size, self.input_size)))
        else:
            self.register_buffer('tril', torch.ones(self.input_size, self.input_size))
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).
        
        Returns:
            The output of a singe attention head,
            of size (batch_size, input_size, output_dim)
        """
        B,T,C = x.size()
        k = F.linear( x, self.key, bias=None) * C**-.5    # [bs, seq_len, qk_dim]
        q = F.linear( x, self.query, bias=None) * C**-.5  # [bs, seq_len, qk_dim]
        v = F.linear( x, self.value, bias=None) * C**-.5  # [bs, seq_len, out_dim] 

        weight = q @ k.transpose(-2,-1) * q.size(-1)**-.5               # [bs, seq_len, seq_len]
        weight = weight.masked_fill(self.tril[:T,:T]==0, float('-inf')) #  //
        weight = F.softmax(weight, dim=-1)                              #  //
        weight = self.dropout(weight)
        out = weight @ v                                                # [bs, seq_len, out_dim]

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple Attention Heads.
    
    Args:
        input_dim: The dimension of input tokens.
        input_size: The (maximal) number of input tokens.
        num_heads: The number of heads.
        out_dim: The dimension of output tokens.
        dropout: The fraction of weights to zero via dropout.
        decoder: True for one-directional attention.
        
    """
    def __init__(
        self, input_dim, input_size, num_heads, out_dim, dropout, decoder=True, rope=False
    ):
        super().__init__()
        assert out_dim % num_heads == 0, "inner dim. must be multiple of num. heads"

        self.input_dim = input_dim
        self.input_size = input_size
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout
        self.decoder = decoder
        self.rope = rope

        self.key=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        self.query=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        self.value=nn.Parameter(
            torch.randn( self.out_dim, self.input_dim)
        )
        self.projection=nn.Parameter(
            torch.randn( self.out_dim, self.out_dim)
        )
        if decoder:
            self.register_buffer('tril', torch.tril(torch.ones(self.input_size, self.input_size)))
        else:
            self.register_buffer('tril', torch.ones(self.input_size, self.input_size))
        if rope:
            self.rotary_emb = RotaryEncoding(self.head_dim)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, input_size, input_dim).
        
        Returns:
            The output of a multi-head attention layer,
            of size (batch_size, input_size, output_dim)
        """
        B,T,C = x.size()
        k = F.linear( x, self.key, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) * C**-.5    # [bs, num_heads, seq_len, head_dim]
        q = F.linear( x, self.query, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) * C**-.5  # [bs, num_heads, seq_len, head_dim]
        v = F.linear( x, self.value, bias=None).view(B, T, self.num_heads, self.head_dim).transpose(1,2) * C**-.5  # [bs, num_heads, seq_len, head_dim]
        if self.rope:
            q, k = self.rotary_emb(q, k)

        weight = q @ k.transpose(-2,-1) * self.head_dim**-.5             # [bs, num_heads, seq_len, seq_len]
        weight = weight.masked_fill(self.tril[:T,:T]==0, float('-inf'))  #  //
        weight = F.softmax(weight, dim=-1)                               #  //
        weight = self.dropout(weight)

        out = (weight @ v).transpose(1,2).reshape(B,T,-1) # [bs, seq_len, out_dim]
        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # out = out.transpose(1,2).contiguous().view(B,T,C)
        out = F.linear( out, self.projection, bias=None) * self.projection.size(-1)**-.5

        return out

    def forward_with_cache(self, x, past_k=None, past_v=None):
        """
        x:      (B, T_new, input_dim)   typically T_new == 1 during generation
        past_k: (B, num_heads, T_past, head_dim) or None
        past_v: same

        returns:
            out:    (B, T_new, out_dim)
            (k,v):  updated caches with shape (B, num_heads, T_total, head_dim)
        """
        B, T_new, C = x.size()

        k = F.linear(x, self.key, bias=None).view(B, T_new, self.num_heads, self.head_dim).transpose(1, 2) * C**-0.5
        q = F.linear(x, self.query, bias=None).view(B, T_new, self.num_heads, self.head_dim).transpose(1, 2) * C**-0.5
        v = F.linear(x, self.value, bias=None).view(B, T_new, self.num_heads, self.head_dim).transpose(1, 2) * C**-0.5
        if self.rope:
            q, k = self.rotary_emb(q, k)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)   # concat on time dimension
            v = torch.cat([past_v, v], dim=2)
        if k.size(2) > self.input_size:  # self.input_size == block_size
            k = k[:, :, -self.input_size:, :]
            v = v[:, :, -self.input_size:, :]

        # q: (B, H, T_new, D), k: (B, H, T_total, D)
        weight = q @ k.transpose(-2, -1) * self.head_dim**-0.5  # (B, H, T_new, T_total)

        # IMPORTANT: in generation we always call this with T_new == 1 and
        # no "future" keys, so we do NOT need a causal mask here.
        weight_softmax = F.softmax(weight, dim=-1)
        self.last_attn = weight_softmax.detach()
        weight = self.dropout(weight_softmax)

        out = (weight @ v).transpose(1, 2).reshape(B, T_new, -1)
        out = F.linear(out, self.projection, bias=None) * self.projection.size(-1)**-0.5

        return out, (k, v)


class FeedForward(nn.Module): #TODO: use mlp from previous arch file? Mean-field vs NTK init?

    def __init__(
        self, input_dim, hidden_dim, output_dim, dropout=0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Decoder Block.
    
    Args:
        embedding_dim: The dimension of the tokens (kept constant past embedding).
        input_size: The (maximal) number of input tokens.
        num_heads: The number of attention heads.
        dropout: The fraction of weights to zero via dropout.
        ffwd_size: Size of the MLP is ffwd_size*embedding_dim.        
    """
    def __init__(
        self, embedding_dim, input_size, num_heads, ffwd_size=4, decoder=True, rope=False, dropout=0
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            input_dim=embedding_dim,
            input_size=input_size, 
            num_heads=num_heads, 
            out_dim=embedding_dim, 
            dropout=dropout,
            decoder=decoder,
            rope=rope,
        )
        #TODO: zero init for biases?
        self.ffwd = FeedForward(embedding_dim, ffwd_size*embedding_dim, embedding_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x

    def forward_with_cache(self, x, layer_past=None):
        """
        x:          (B, T_new, C)
        layer_past: None or (k, v) for this layer

        returns:
            x:            (B, T_new, C)
            layer_present: (k, v) with all keys/values up to now
        """
        if layer_past is None:
            past_k, past_v = None, None
        else:
            past_k, past_v = layer_past

        y, (k, v) = self.attn.forward_with_cache(self.ln1(x), past_k=past_k, past_v=past_v)
        x = x + self.dropout(y)
        x = x + self.dropout(self.ffwd(self.ln2(x)))
        return x, (k, v)


class CLM(nn.Module):
    """
    Causal (decoder-only) Language Model.
    
    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.
        dropout: The fraction of weights to zero via dropout.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, ffwd_size, num_layers, dropout, rope=False, share_emb=True
    ):
        super().__init__()

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ffwd_size = ffwd_size
        self.num_layers = num_layers
        self.rope = rope

        self.token_embedding_table = nn.Embedding(vocab_size, self.embedding_dim)
        # TODO: flag for APE?
        if not rope:
            self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                    ffwd_size=self.ffwd_size,
                    decoder=True,
                    rope=rope,
                    dropout=dropout
                ) for _ in range(self.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size)
        if share_emb:
            self.lm_head.weight = self.token_embedding_table.weight


    def forward(self, idx, targets=None):

        B,T = idx.size()

        if self.rope:
            x = self.token_embedding_table(idx) # [bs, seq_len, embedding_dim]
        else:
            token_emb = self.token_embedding_table(idx) # [bs, seq_len, embedding_dim]
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # [seq_len, embedding_dim]
            x = token_emb + pos_emb  # [bs, seq_len, embedding_dim]

        x = self.blocks(x)       # [bs, seq_len, embedding_dim]
        x = self.ln_f(x)         # [bs, seq_len, embedding_dim]
        logits = self.lm_head(x) # [bs, seq_len, input_dim]

        return logits

    def forward_with_cache(self, idx, cache=None):  # , past_len=0):
        """
        idx:      (B, T_new) token ids for the *new* tokens
        cache:    None or list of (k, v) for each layer
        past_len: how many tokens came before these new ones (for positions)

        returns:
            logits:    (B, T_new, vocab)
            new_cache: list[(k, v)] for each layer
        """
        B, T_new = idx.size()
        device = idx.device

        token_emb = self.token_embedding_table(idx)  # (B, T_new, C)
        # Derive how many tokens we've already cached from the first layer
        if cache is None:
            T_past = 0
        else:
            past_k, _ = cache[0]          # past_k: (B, num_heads, T_past, head_dim)
            T_past = past_k.size(2)

        # absolute positions for these tokens: [past_len, ..., past_len + T_new - 1]
        # pos_idx = torch.arange(T_past, T_past + T_new, device=device)
        pos_idx = torch.arange(T_past, T_past + T_new, device=device) % self.block_size

        assert pos_idx.max().item() < self.block_size, \
                    f"pos_idx {pos_idx.max().item()} >= block_size {self.block_size}"

        if self.rope:
            x = token_emb
        else:
            pos_emb = self.position_embedding_table(pos_idx)  # (T_new, C)
            x = token_emb + pos_emb  # broadcasting over batch

        new_cache = []
        # manually iterate over blocks to thread the per-layer cache
        for layer_idx, block in enumerate(self.blocks):
            layer_past = None if cache is None else cache[layer_idx]
            x, layer_present = block.forward_with_cache(x, layer_past)
            new_cache.append(layer_present)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_cache

    def generate(self, idx, num_tokens):

        for _ in range(num_tokens):
            # get the prediction
            idx_cond = idx[:,-self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # (bs, input_dim)
            probs = F.softmax(logits, dim=-1) # (bs, input_dim)
            #TODO: possibly restrict probs to the top k values, might be useful for large vocab sizes
            idx_next = torch.multinomial(probs, num_samples=1) # (bs, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (bs, seq_len+1)

        return idx


    def generate_fast(self, idx, eos_id, max_new_tokens=2_048):
        device = idx.device
        B, T0 = idx.shape
        cache = None
        with torch.inference_mode():
            logits, cache = self.forward_with_cache(idx, cache=cache)
            logits_step = logits[:, -1, :]  # logits for last prompt token
        logp_steps  = []   # will store (B, 1) log-probs
		# 2) Sample first continuation token
        logprobs = F.log_softmax(logits_step, dim=-1)
        idx_next = torch.multinomial(logprobs.exp(), num_samples=1)  # (B, 1)
        idx = torch.cat([idx, idx_next], dim=1)
        seen_eos = (idx_next.squeeze(-1) == eos_id)
        step_logp = logprobs.gather(-1, idx_next)     # (B, 1)
        logp_steps.append(step_logp)
        for step in range(max_new_tokens - 1):
            if seen_eos.all():
                break
            idx_step = idx[:, -1:].contiguous()  # (B, 1) last token
            with torch.inference_mode():
                logits, cache = self.forward_with_cache(idx_step, cache=cache)
                # logits is for that single position, shape (B, 1, vocab)
                logits_step = logits[:, -1, :]  # (B, vocab)
            logprobs = F.log_softmax(logits_step, dim=-1)
            idx_next = torch.multinomial(logprobs.exp(), num_samples=1)  # (B, 1)
            # gather log p of the chosen token
            step_logp = logprobs.gather(-1, idx_next)     # (B, 1)
            logp_steps.append(step_logp)
            idx = torch.cat([idx, idx_next], dim=1)   # extend sequence
            seen_eos |= (idx_next.squeeze(-1) == eos_id)
            logp = torch.cat(logp_steps, dim=1)

        # device = idx.device
        # B, T0 = idx.shape

        # logp_steps  = []   # will store (B, 1) log-probs
        # seen_eos = torch.zeros(B, dtype=bool, device=device)

        # for _ in tqdm(range(max_new_tokens)):

        #     if seen_eos.all():
        #         break
        #
        #     # NOTE: conditioning only on the last `block_size` tokens (since model was only trained
        #     # on this max context, no RoPE embeddings or anything like that)
        #     idx_cond = idx[:, -self.block_size:]
        #     logits = self(idx_cond)                       # (B, T_cond, vocab)
        #     logits = logits[:, -1, :]                     # (B, vocab)

        #     logprobs = F.log_softmax(logits, dim=-1)      # (B, vocab)

        #     # Sample next token for *all* sequences
        #     idx_next = torch.multinomial(logprobs.exp(), num_samples=1)  # (B, 1)

        #     # gather log p of the chosen token
        #     step_logp = logprobs.gather(-1, idx_next)     # (B, 1)
        #     logp_steps.append(step_logp)

        #     idx = torch.cat((idx, idx_next), dim=1)       # (B, T0 + step + 1)

        #     seen_eos |= (idx_next.squeeze(-1) == eos_id)

        #     logp = torch.cat(logp_steps, dim=1)

        # Now truncate each story at its *first* EOS in the generated part
        stories = []
        story_logps = []

        for b in range(B):
            tokens = idx[b]                  # length T0 + num_steps_run
            generated = tokens[T0:]          # generated suffix

            eos_pos = (generated == eos_id).nonzero(as_tuple=False)

            if eos_pos.numel() > 0:
                gen_len = eos_pos[0].item() + 1  # include EOS
            else:
                gen_len = generated.size(0)      # no EOS -> keep everything

            # Full sequence: prompt + generated up to gen_len
            story_tokens = tokens[:T0 + gen_len]
            stories.append(story_tokens)

            # Log probs: just the generated tokens up to gen_len
            story_logp = logp[b, :gen_len]
            story_logps.append(story_logp)

        return stories, story_logps

class BLM(nn.Module):
    """
    Bidirectional (BERT-like) Language Model.
    
    Args:
        vocab_size: The dimension of input tokens.
        block_size: The (maximal) number of input tokens.
        embedding_dim: The embedding dimension.
        num_heads: The number of attention heads.
        num_layers: The number of layers.

        dropout: The fraction of weights to zero via dropout.
    """
    def __init__(
        self, vocab_size, block_size, embedding_dim, num_heads, num_layers, dropout
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.token_embedding_table = nn.Embedding(vocab_size+1, self.embedding_dim)
        self.position_embedding_table = nn.Embedding(self.block_size, self.embedding_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=self.embedding_dim,
                    input_size=self.block_size,
                    num_heads=self.num_heads,
                    ffwd_size=4,
                    decoder=False,
                    dropout=dropout
                ) for _ in range(self.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.embedding_dim)
        self.lm_head = nn.Linear(self.embedding_dim, vocab_size)


    def forward(self, idx):

        B,T = idx.size() # takes [bs, seq_len] input with elements in range(vocab_size)

        token_emb = self.token_embedding_table(idx) # [bs, seq_len, embedding_dim]
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # [seq_len, embedding_dim]
        x = token_emb + pos_emb  # [bs, seq_len, embedding_dim]

        x = self.blocks(x)       # [bs, seq_len, embedding_dim]
        x = self.ln_f(x)         # [bs, seq_len, embedding_dim]
        logits = self.lm_head(x[:,-1,:]) # [bs, vocab_size]

        return logits
