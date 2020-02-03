import nltk
import json
from pathlib import Path
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np


class MultiwozSet():
    """Interface to the cleaned MultiWOZ dataset"""
    def __init__(self, path=None, obj=None):
        if path:
            self.data = MultiwozSet._load(path=path)
        else:
            self.data = MultiwozSet._load(obj=obj)
    
    @staticmethod
    def _load(**kwargs):
        pass

class Vocab(MultiwozSet):
    def __init__(self):
        super().__init__()
        self.word_to_id = {}
        self.id_to_word = {}

    def construct_from_data(self):
        self.word_to_id['<pad>'] = 0   # Pad Token
        self.word_to_id['<s>'] = 1 # Start Token
        self.word_to_id['</s>'] = 2    # End Token
        self.word_to_id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word_to_id['<unk>']

        #Fill with all token from all utterances

    def get(self):
        return self.word_to_id

    def __len__(self):
        return len(self.word_to_id)

class CandidateSet():
    """Models the candidate set for a single user utterance"""
    def __init__(self, utterance):
        self.unigrams = utterance.split()
        self.bigrams = list(nltk.bigrams(self.unigrams))

    def get_unigrams(self):
        return self.unigrams

    def get_bigrams(self):
        return self.bigrams

class SlotSet(MultiwozSet):
    """Models the slot type set presented in Section 3 of Budzianowski et al. (MultiWOZ - A Large-Scale
    Multi-Domain..."""

    def __init__(self,path=None, obj=None):
        super().__init__()
        if path:
            self.data = self._load(path)
        else:
            self.data = self._load(obj)
        self.slot_set = None

    def get(self):
        return self.slot_set 

    def __len__(self):
        return len(self.slot_set)
