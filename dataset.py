import nltk
import json
import os
from pathlib import Path
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np


DATA = Path.cwd().parent / 'data'

DOMAINS = ['attraction','hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train'] 


def read_dials(base, domains, word_to_id, usr=True):
    """
    Reads all dials into vocab dict and saves to json
    
    --base: Path object
    --domains: (list) dialogue domains
    --word_to_id: (dict) existing word to index mappings
    --usr: whether to get usr or system transcript
    """
    trans_key = 'transcript' if usr else 'system_transcript'
    out_ext = 'usr_vocab.json' if usr else 'system_vocab.json'
    i = len(word_to_id)
    for d in domains:
        for dial in json.load(open(base / '{}.json'.format(d), 'r')):
            for bs in dial['dialogue']:
                for w in bs.get(trans_key):
                    if not word_to_id.get(w):
                        word_to_id[w] = i
                        i += 1

    json.dump(word_to_id, open(DATA / out_ext, 'w'))
            

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
