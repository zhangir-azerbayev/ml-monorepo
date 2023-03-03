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
