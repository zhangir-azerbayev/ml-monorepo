import sys
from functools import partial, reduce 
import argparse

from omegaconf import OmegaConf

import torch 
import torch.nn as nn
from torch.nn import functional as F 
from model import *
torch.manual_seed(1337)

import wandb

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

def train(model, train_loader, val_loader, config, device, log=True):
    device = config.device
    learning_rate = config.train.learning_rate
    max_iters = config.train.max_iters
    eval_interval = config.train.eval_interval
    eval_iters = config.train.eval_iters
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(max_iters): 
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, eval_iters)
            log_dict = {"train_loss": losses['train'], "val_loss": losses['val']}
            if log:
                wandb.log(data=log_dict, step=i)
            print(f"step {i}", log_dict)
        
        inputs, targets = train_loader.get_batch()

        logits, loss = model(inputs.to(device), targets.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Training character LMs")
    parser.add_argument('config')
    args = parser.parse_args()
    config_path = args.config

    config = OmegaConf.load(config_path)
    batch_size = config.train.batch_size
    context_length = config.model.context_length
    
    if torch.cuda.is_available():
        device = config.device
    else:
        device="cpu"

    try: 
        run_obj = wandb.init(project="shakespeare_characters", name=config.wandb.name)
        log=True
    except AttributeError: 
        print("NOT LOGGING TO WANDB")
        log=False

    with open("data/shakespeare.txt") as f: 
        text = f.read()

    print(f"EXAMPLE TEXT: {text[:100]}")
 
    encode, decode, vocab_size = build_tokenizer(text)

    match config.model.arch:
        case "BigramModel":
            model = BigramModel(vocab_size).to(device)
        case "SingleHeadModel": 
            model = SingleHeadModel(context_length, 
                                    config.model.d_model, 
                                    vocab_size
                                    ).to(device)
        case "MultiHeadModel": 
            model = MultiHeadModel(context_length, 
                                   config.model.d_model, 
                                   config.model.num_heads, 
                                   vocab_size
                                   ).to(device)
        case "SingleLayerModel":
            model = SingleLayerModel(context_length, 
                                     config.model.d_model, 
                                     config.model.num_heads,
                                     vocab_size).to(device)
        case "ParallelSingleLayerModel":
            model = ParallelSingleLayerModel(context_length, 
                                     config.model.d_model, 
                                     config.model.num_heads,
                                     vocab_size).to(device)

        case "GPTModel": 
            model = GPTModel(context_length, 
                             config.model.d_model, 
                             config.model.num_heads, 
                             config.model.n_layers, 
                             vocab_size).to(device)
        case "ParallelGPTModel":
            model = ParallelGPTModel(context_length, 
                             config.model.d_model, 
                             config.model.num_heads, 
                             config.model.n_layers, 
                             vocab_size).to(device)
        case _: 
            raise ValueError("config.model.arch invalid")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NUM PARAMS: {n_params}")

    prompt = text[:1]
    outs = model.generate(torch.unsqueeze(encode(prompt), dim=0).to(device), max_new_tokens=100)
    out_text = decode(torch.squeeze(outs))
    print("before training: ", out_text)

    # create dataloaders that return tokens
    tokens = encode(text).to(device)
    n = int(0.9*len(tokens))
    try:
        train_set_length = config.train.train_set_length
        print(f"TRUNCATING TRAIN SET TO {train_set_length} characters")
        train_loader = DataLoader(tokens[:train_set_length], 
                context_length, batch_size
                )
    except AttributeError:
        print("USING FULL TRAINING SET")
        train_loader = DataLoader(tokens[:n], context_length, batch_size)
    val_loader = DataLoader(tokens[n:], context_length, batch_size)

    model = train(model, train_loader, val_loader, config, device, log=log)

    outs = model.generate(torch.unsqueeze(encode(prompt), dim=0).to(device), 
            max_new_tokens=500
            )
    out_text = decode(torch.squeeze(outs))
    print("after training: ", out_text)

if __name__=="__main__": 
    main()
