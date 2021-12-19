import torch

import path
from data import TAG_MAP, condtraints, build_dict, read_corpus

corpus = read_corpus(path.train_file_path)
dct = build_dict(corpus)

BATCH_SIZE = 32


class Config:
    condtraints = condtraints
    name = "hidden_256_embed_150"
    hidden_size = 256
    num_tags = len(TAG_MAP)
    embed_dim = 300
    dropout = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = len(dct)
