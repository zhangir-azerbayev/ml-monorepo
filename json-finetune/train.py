import json
import ndjson
from collections import OrderedDict
from typing import Dict

import argparse
from omegaconf import OmegaConf

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer

@dataclass 
class Config: 
    model: str
    
    train_path: str
    keys: List[str]

    lr: float 
    train_steps: int
    batch_size: int

    save_dir: str

def serialize(
        example: Dict[str, str], 
        keys: List[str]
):
    ordered_example = OrderedDict((k, example[k]) for k in keys)
    return json.dumps(ordered_example, indent=2)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        toks = tokenizer(x, padding='max_length', truncate=True, return_tensors='pt') 
        return {k: toks[k].to('cuda') for k in toks}

def data_collator(data): 
    return {
            'input_ids': torch.stack([x['input_ids'] for x in data]),
            'attention_mask': torch.stack([x['attention_mask'] for x in data]),
            'labels': torch.stack([x['input_ids'] for x in data])
    }

def main(conf):
    model = AutoModelForCausalLM.from_pretrained(conf.model).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(conf.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'

    with open(conf.train_path, "w") as f: 
        ds = ndjson.load(f)

    train_texts = [serialize(x, conf.keys) for x in ds]
    train_tokens = [
            tokenizer(x, padding='max_length', truncate=True, return_tensors='pt') 
            for x in train_texts
            ] 

    training_args = TrainingArguments(
            output_dir=conf.save_dir, 
            do_train='true', 
            per_device_train_batch_size=conf.batch_size,
            learning_rate=conf.lr, 
            max_steps=conf.train_steps, 
            warmup_steps=conf.warmup_steps,
            weight_decay=0.01
            logging_steps=conf.log_interval,
            save_steps=conf.train_steps, 
            fp16=True
    )

    Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokens,
    ).train()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Path to yaml config file')
    args = parser.parse_args()

    schema = OmegaConf.from_structured(Config)
    values = OmegaConf.load(args.config)

    conf = OmegaConf.merge(schema, conf)

    main(conf)    
