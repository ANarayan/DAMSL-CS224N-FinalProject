import torch
import torch.nn as nn
import numpy as np
from dataset import Vocab

class VocabEmbeddings(nn.Module):
    """Lookup of embeddings for all tokens in the train/valid/test vocabulary"""
    def __init__(self, emb_dim):
        super().__init__()
        self.vocab = Vocab()
        self.embeddings = nn.Embeddings(len(self.vocab), emb_dim)
