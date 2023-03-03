import sys
from functools import partial, reduce 

from omegaconf import OmegaConf

import torch 
import torch.nn as nn
from torch.nn import functional as F 
from model import BigramModel
torch.manual_seed(1337)

def build_tokenizer(text): 
    print("hit build_tokenizer method!")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("vocab: ", ' '.join(chars))
    print(f"vocab size: {vocab_size}")

    int_of_char = {ch:i for i, ch in enumerate(chars)}
    char_of_int = {i:ch for i, ch in enumerate(chars)}

    encode = lambda text: torch.tensor([int_of_char[ch] for ch in text])
    decode = lambda toks: reduce(lambda x, y: x + y, 
                                 [char_of_int[i.item()] for i in toks]
                                )

    print(decode(encode("the tokenizer works properly!")))

    return encode, decode, vocab_size

class DataLoader(): 
    def __init__(self, tokens, context_length, batch_size):
        self.tokens = tokens
        self.context_length = context_length
        self.batch_size = batch_size

    def get_batch(self):
        idxs = torch.randint(len(self.tokens)-self.context_length, (self.batch_size,))

        x = torch.stack([self.tokens[i:i+self.context_length] for i in idxs])
        y = torch.stack([self.tokens[i+1:i+self.context_length+1] for i in idxs])
        return x, y

def estimate_loss(model, train_loader, val_loader, eval_iters):
    out = {}
    model.eval()
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): 
            inputs, targets = loader.get_batch()
            logits, loss = model(inputs, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(model, train_loader, val_loader, config):
    device = config.device
    learning_rate = config.train.learning_rate
    max_iters = config.train.max_iters
    eval_interval = config.train.eval_interval
    eval_iters = config.train.eval_iters
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(max_iters): 
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters)
            print(f"step {i}, ", f"train loss {losses['train']:.4f}, ", 
                    f"val loss {losses['val']:.4f}")
        
        inputs, targets = train_loader.get_batch()

        logits, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model

def main():
    config = OmegaConf.load("configs/default.yaml")
    batch_size = config.train.batch_size
    context_length = config.model.context_length

    with open("data/shakespeare.txt") as f: 
        text = f.read()
    
    encode, decode, vocab_size = build_tokenizer(text)

    model = BigramModel(vocab_size)
    
    prompt = "LORD BANQUO"
    outs = model.generate(torch.unsqueeze(encode(prompt), dim=0), max_new_tokens=100)
    out_text = decode(torch.squeeze(outs))
    print("before training: ", out_text)

    tokens = encode(text)
    n = int(0.9*len(tokens))
    train_loader = DataLoader(tokens[:n], context_length, batch_size)
    val_loader = DataLoader(tokens[n:], context_length, batch_size)

    model = train(model, train_loader, val_loader, config)

    outs = model.generate(torch.unsqueeze(encode(prompt), dim=0), max_new_tokens=100)
    out_text = decode(torch.squeeze(outs))
    print("after training: ", out_text)

if __name__=="__main__": 
    main()
