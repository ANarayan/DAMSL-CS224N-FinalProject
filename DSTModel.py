import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

from embeddings import VocabEmbeddings


class DST(nn.Module):
    def __init__(self):
        super(DST, self).__init__()

    def forward(self, x):
        pass 

class SentenceBiLSTM(nn.Module): 
    def __init__(self, hidden_dim, embed_dim, candidate_encoder, batch_size):
        super(SentenceBiLSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim= hidden_dim
        self.embedding_dim= embed_dim

        # candidate encoder is an embedding lookup of dimensions embed_dim
        self.candidate_encoder = candidate_encoder
        
        # Initialized the biLSTM
        # input: embedded word representation of dim embed_size
        # output: sentence representation of dim hidden_size
        self.sentence_biLSTM = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))

    def forward(self, sentence):
        embeds = self.candidate_encoder(sentence).view(len(sentence), self.batch_size, -1)
        encoding, self.hidden = self.sentence_biLSTM(embeds, self.hidden)
        return self.hidden

class HierarchicalLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super(HierarchicalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hierarchical_lstm = nn.LSTM(self.input_size, self.hidden_size) 

    def forward(self, x):
        pass

class DialogueActsLSTM(nn.Module):
    def __init__(self):
        super(DialogueActsLSTM, self).__init__()

    def forward(self, x):
        pass

class ClassificationNet(nn.Module):
    def __init__(self, context_dim, hidden1_dim, hidden2_dim):
        super(ClassificationNet, self).__init__()
        self.context_feature_dim = context_dim
        self.linear1_dim = hidden1_dim
        self.linear2_dim = hidden2_dim

        self.fc1 = nn.Linear(self.context_feature_dim, self.linear1_dim)
        self.fc2 = nn.Linear(self.linear1_dim, self.linear2_dim)
        self.fc3 = nn.Linear(self.linear2_dim, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return F.sigmoid(output)



        

