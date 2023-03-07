Transformers, from scratch. Trained on Andrej Karpathy's [tiny
Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), following his
Youtube videos. 

Works up to a 23 million parameter transformer. 

I also implement versions of the transformers that use [QK normalization, along with parallel MLP and attention layers](https://arxiv.org/abs/2302.05442). I observe no performance degredation, suggesting that these methods are at worst harmless. 

Logs are public: [wandb](https://wandb.ai/math-lm/shakespeare_characters/overview?workspace=user-zhangir-azerbayev)
