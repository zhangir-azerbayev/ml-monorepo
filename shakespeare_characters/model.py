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

class Head(nn.Module): 
    """single self-attention head"""
    def __init__(self, context_length, n_embed, head_size, dropout=.2): 
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    
        self.context_length = context_length

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """should handle T <= context_length
        inputs: (B, T, n_embed)

        out: (B, T, head_size)
        """
        # inputs is (B, T, C=n_embed)
        B, T, C = inputs.shape
        assert T <= self.context_length

        keys = self.key(inputs) # (B, T, C=head_size)
        queries = self.query(inputs) # (B, T, C)
        values = self.query(inputs) # (B, T, C)

        # (B, T, T) = (B, T, C) @ (B, C, T)         
        attn_logits = queries @ keys.transpose(-2, -1) * keys.shape[-1]**-.5

        attn_logits = attn_logits.masked_fill(self.tril[:T, :T]==0,  float('-inf'))
        attn_scores = F.softmax(attn_logits, dim=-1)
        attn_scores = self.dropout(attn_scores)
        
        # (B, T, C) = (B, T, T) @ (B, T, C)
        # remember C=head_size
        outs = attn_scores @ values 

        return outs

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
    def __init__(self, context_length, n_embed, head_size, vocab_size,
            dropout=.2): 
        super().__init__(context_length)

        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(context_length, n_embed)
        self.attention = Head(context_length, n_embed, head_size)
        self.proj = nn.Linear(head_size, vocab_size, bias=False)

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(torch.arange(T)) # (B, T, C=n_embed)
        outs = self.attention(embeds) # (B, T, C=head_size)
        logits = self.proj(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class TwoLayerSingleHeadModel(AttentionModel): 
    def __init__(self, context_length, n_embed, head_size, vocab_size,
            dropout=.2): 
        super().__init__(context_length)

        self.embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(context_length, n_embed)
        self.attention1 = Head(context_length, n_embed, head_size)
        self.layernorm = nn.LayerNorm(head_size)
        self.attention2 = Head(context_length, head_size, head_size)
        self.proj = nn.Linear(head_size, vocab_size, bias=False)

    def forward(self, inputs, targets=None): 
        """
        input: (B, T), where T<= vocab_size

        out: (B, T, vocab_size), (B, T, vocab_size)
        """
        B, T = inputs.shape
        
        # note the broadcasting below
        embeds = self.embed(inputs) + self.pos_embed(torch.arange(T)) # (B, T, C=n_embed)
        outs = self.attention1(embeds) # (B, T, C=head_size)
        outs = self.layernorm(outs)
        outs = self.attention2(outs)
        logits = self.proj(outs)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
