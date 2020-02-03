import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np


class DST(nn.Module):
    def __init__(self):
        super(DST, self).__init__()

    def forward(self, x):
        pass 

class SentenceBiLSTM(nn.Module): 
    def __init__(self):
        super(SentenceBiLSTM, self).__init__()

    def forward(self, x):
        pass

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
    def __init__(self):
        super(ClassificationNet, self).__init__()

    def forward(self, x):
        pass


        

