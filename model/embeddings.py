import torch
import torch.nn as nn
import numpy as np
#from dataset import Vocab

class Embeddings(nn.Module):
    """Lookup of embeddings for all tokens in the train/valid/test vocabulary"""
    def __init__(self, emb_dim, vocab):
        super().__init__()
        self.embeddings = nn.Embedding(len(vocab), emb_dim)

