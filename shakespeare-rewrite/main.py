import logging
import sys
import yaml
from typing import Literal

"""
 __       __            ___    __  
|__) \ / |  \  /\  |\ |  |  | /  ` 
|     |  |__/ /~~\ | \|  |  | \__, 
                                   
Pydantic is the most widely used data validation library for Python.
"""
from pydantic import BaseModel, ValidationError, model_validator


"""

 __         ___  __        __   __        ___ 
/  \  |\/| |__  / _`  /\  /  ` /  \ |\ | |__  
\__/  |  | |___ \__> /~~\ \__, \__/ | \| |    
                                              
OmegaConf is a YAML based hierarchical configuration system, with support for 
merging configurations from multiple sources (files, CLI argument, environment 
variables)
"""
from omegaconf import OmegaConf

class TrainingConfig(BaseModel):
    """
    Training and optimizer hyperparameters.
    """
    steps: int = 1000
    batch_size: int = 8 

    optimizer: Literal['SGD', 'Adam', 'Adafactor'] = 'Adafactor'
    lr: float = 3e-4

class ModelConfig(BaseModel):
    """
    Architecture hyperparameters for the transformer model
    """
    seq_len: int = 16

    pos_embed: Literal['learned', 'none'] = 'none'

    d_model: int = 32
    n_heads: int = Field(
            default=4 
            description="Must divide `d_model`"
            )
    mlp_fan: int = 4
    activation: Literal['relu', 'gelu'] = 'gelu'

    n_layers: int = 4
 
    @model_validator(mode='after')
    def heads_div_model(self):
        if self.d_model % self.n_heads != 0:
            raise ValidationError('n_heads must divide d_model')

class Config(TrainingConfig, ModelConfig): 
    pass

def main(config):
    logging.info(config)
    # do machine learning stuff here

if __name__=="__main__": 
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    cli_cfg = OmegaConf.from_cli()

    if 'config_path' in cli_cfg:
        cfg = OmegaConf.load(cli_cfg.config_path)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    else: 
        cfg = cli_cfg

    print(cfg)

    config = Config(**cfg)
    main(config)
