import sys

import torch 
import torch.nn as nn
from torch.nn import functional as F 
torch.manual_seed(1337)

from functools import partial, reduce 

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        # x is (B, T)
        logits = self.embed(x) # (B, T, vocab_size)
        probs = F.softmax(logits, dim=-1) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, seqs, max_new_tokens):
        self.eval()
        # seqs is (B, T)
        for _ in range(max_new_tokens):
            # crop prompt to last token
            seqs_eff = seqs[:, -1:] # (B, 1)
            # print(f"seqs eff shape {seqs_eff.shape}")

            logits, _ = self(seqs_eff) # (B, 1, vocab_size)

            # unsqueeze
            logits = logits[:, -1, :]
            # print(f"logits shape {logits.shape}")
            probs = F.softmax(logits, dim=-1)
            # print(f"probs shape {probs.shape}")

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            seqs = torch.cat((seqs, idx_next), dim=-1) # (B, T+1)

        self.train()
        return seqs

class MLP(nn.Module):
    def __init__(self, d_model, fan, dropout=.2):
        super().__init__()
        self.fc = nn.Linear(d_model, fan*d_model)
        self.nonlin = nn.GELU()
        self.proj = nn.Linear(fan*d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (*, d_model)

        out: (*, d_model)
        """
        x = self.fc(x)
        x = self.dropout(x)
        x = self.nonlin(x)
        x = self.proj(x)
        return x


class Head(nn.Module): 
    """single self-attention head"""
    def __init__(self, context_length, d_model): 
        super().__init__()
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    
        self.context_length = context_length

    def forward(self, inputs):
        """should handle T <= context_length
        inputs: (B, T, d_model)

        out: (B, T, d_model)
        """
        # inputs is (B, T, C=d_model)
        B, T, C = inputs.shape
        assert T <= self.context_length
        
        # Note: this is inefficient. We fix this in MulitHead
        keys = self.key(inputs) # (B, T, C=d_model)
        queries = self.query(inputs) # (B, T, C)
        values = self.query(inputs) # (B, T, C)

        # (B, T, T) = (B, T, C) @ (B, C, T)         
        attn_logits = queries @ keys.transpose(-2, -1) * keys.shape[-1]**-.5

        attn_logits = attn_logits.masked_fill(self.tril[:T, :T]==0,  float('-inf'))
        attn_scores = F.softmax(attn_logits, dim=-1)
 
        # (B, T, C) = (B, T, T) @ (B, T, C)
        # remember C=d_model
        
        outs = attn_scores @ values 

        return outs

class MultiHead(nn.Module):
    def __init__(self, context_length, d_model, num_heads):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
 
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads==0
        self.head_size = d_model//num_heads

    def forward(self, inputs):
        """
        inputs: (B, T, C=d_model)

        out: (B, T, C=d_model)
        """
        B, T, C = inputs.shape # C = d_model
        assert T<=self.context_length


        qkv_mat = self.qkv(inputs) # (B, T, C=3*head_size*num_heads)

        projs = qkv_mat.view(B, T, self.num_heads, 3*self.head_size)
        projs = projs.transpose(1, 2) # (B, num_heads, T, 3*head_size)
        assert (B, self.num_heads, T, 3*self.head_size) == projs.shape

        q, k, v = torch.split(projs, self.head_size, dim=3) # (B, num_heads, T, head_size)
        assert q.shape == v.shape and (B, self.num_heads, T, self.head_size) == q.shape

        # attention
        # (B, num_heads, T, T) = (B, num_heads, T, head_size) @ (B, num_heads, head_size, T)
        logits = q @ k.transpose(-1, -2) * k.shape[-1]**-.5 
        logits = logits.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        scores = F.softmax(logits, dim=-1)
        assert (B, self.num_heads, T, T) == scores.shape
         
        # (B, num_heads, T, head_size) = (B, nh, T, T)@(B, nh, T, head_size)
        outs = scores @ v 

        outs = outs.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return outs

class TransformerBlock(nn.Module):
    def __init__(self, context_length, d_model, num_heads, fan=4): 
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = MultiHead(context_length, d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = MLP(d_model, fan)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ParallelTransformerBlock(nn.Module): 
    def __init__(self, context_length, d_model, num_heads, fan=4): 
        super().__init__()
        self.qkv = nn.Linear(d_model, 3*d_model + fan*d_model)
        self.mlp_out = nn.Linear(fan*d_model, d_model)
        
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln_q = nn.LayerNorm(d_model//num_heads, elementwise_affine=False)
        self.ln_k = nn.LayerNorm(d_model//num_heads, elementwise_affine=False)
        self.gelu = nn.GELU()

        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
 
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.fan = fan

        assert d_model % num_heads==0
        self.head_size = d_model//num_heads

    def forward(self, inputs):
        """
        inputs: (B, T, C=d_model)

        out: (B, T, C=d_model)
        """
        B, T, C = inputs.shape # C = d_model
        assert T<=self.context_length

        qkv_hidden_mat = self.qkv(self.ln(inputs)) # (B, T, C=3*d_model + self.fan*d_model)
        assert (B,T,3*self.d_model+self.fan*self.d_model) == qkv_hidden_mat.shape
        
        split_lst = 3*[self.d_model] + [self.fan*self.d_model]
        q, k, v, hidden = torch.split(qkv_hidden_mat, split_lst, dim=-1)
        assert (B, T, self.d_model) == q.shape
        assert (B, T, self.fan*self.d_model) == hidden.shape

        q = self.ln_q(q.view(B, T, self.num_heads, self.head_size).transpose(1,2))
        k = self.ln_k(k.view(B, T, self.num_heads, self.head_size).transpose(1,2))
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1,2)

        mlp_out = self.mlp_out(self.gelu(hidden))
        assert (B, T, self.d_model) == mlp_out.shape

        # attention
        # (B, num_heads, T, T) = (B, num_heads, T, head_size) @ (B, num_heads, head_size, T)
        logits = q @ k.transpose(-1, -2) * k.shape[-1]**-.5 

        logits = logits.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        scores = F.softmax(logits, dim=-1)
        assert (B, self.num_heads, T, T) == scores.shape
         
        # (B, num_heads, T, head_size) = (B, nh, T, T)@(B, nh, T, head_size)
        outs = scores @ v 

        outs = outs.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return outs + mlp_out + inputs

class AttentionModel(nn.Module): 
    def __init__(self, context_length): 
        super().__init__()
        self.context_length = context_length

    def forward(self, inputs, targets=None): 
        raise NotImplementedError 

    @torch.no_grad()
    def generate(self, seqs, max_new_tokens):
        self.eval()
        # seqs is (B, T)
        for _ in range(max_new_tokens):
            # crop prompt to context_length
            seqs_eff = seqs[:, -self.context_length:] # (B, T)
            # print(f"seqs eff shape {seqs_eff.shape}")

            logits, _ = self(seqs_eff) # (B, 1, vocab_size)

            # only care about T
            logits = logits[:, -1, :]
            # print(f"logits shape {logits.shape}")
            probs = F.softmax(logits, dim=-1)
            # print(f"probs shape {probs.shape}")

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            seqs = torch.cat((seqs, idx_next), dim=-1) # (B, T+1)

        self.train()
        return seqs

class SingleHeadModel(AttentionModel): 
    def __init__(self, context_length, d_model, vocab_size,
            dropout=.2): 
        super().__init__(context_length)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.attention = Head(context_length, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer('arange', torch.arange(context_length))

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(self.arange[:T]) # (B, T, C=d_model)
        outs = self.attention(embeds) # (B, T, C=head_size)
        logits = self.lm_head(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class MultiHeadModel(AttentionModel): 
    def __init__(self, context_length, d_model, num_heads, vocab_size): 
        super().__init__(context_length)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.attention = MultiHead(context_length, d_model, num_heads)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer('arange', torch.arange(context_length))

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(self.arange[:T]) # (B, T, C=d_model)
        outs = self.attention(embeds) # (B, T, C=head_size*num_heads)
        logits = self.lm_head(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class SingleLayerModel(AttentionModel):
    def __init__(self, context_length, d_model, num_heads, vocab_size):
        super().__init__(context_length)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.block = TransformerBlock(context_length, d_model, num_heads)
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer('arange', torch.arange(context_length))

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(self.arange[:T]) # (B, T, C=d_model)
        outs = self.ln(self.block(embeds)) # (B, T, C=d_model)
        logits = self.lm_head(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class ParallelSingleLayerModel(AttentionModel):
    def __init__(self, context_length, d_model, num_heads, vocab_size):
        super().__init__(context_length)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.block = ParallelTransformerBlock(context_length, d_model, num_heads)
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer('arange', torch.arange(context_length))

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(self.arange[:T]) # (B, T, C=d_model)
        outs = self.ln(self.block(embeds)) # (B, T, C=d_model)
        logits = self.lm_head(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class GPTModel(AttentionModel):
    def __init__(self, context_length, d_model, num_heads, n_layers, vocab_size):
        super().__init__(context_length)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(context_length, d_model, num_heads)
                for _ in range(n_layers)]
                )
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer('arange', torch.arange(context_length))

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(self.arange[:T]) # (B, T, C=d_model)
        outs = self.ln(self.blocks(embeds)) # (B, T, C=d_model)
        logits = self.lm_head(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

class ParallelGPTModel(AttentionModel):
    def __init__(self, context_length, d_model, num_heads, n_layers, vocab_size):
        super().__init__(context_length)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[ParallelTransformerBlock(
            context_length, d_model, num_heads)
                for _ in range(n_layers)]
                )
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer('arange', torch.arange(context_length))

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(self.arange[:T]) # (B, T, C=d_model)
        outs = self.ln(self.blocks(embeds)) # (B, T, C=d_model)
        logits = self.lm_head(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
