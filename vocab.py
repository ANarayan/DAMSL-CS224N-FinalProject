import os
import nltk
import json
import pickle as pkl
import logging
import torch
from argparse import ArgumentParser
from pathlib import Path
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np
from utils import pad

#Default dir 'formatted_data' in the parent dir of repo
DATA_DIR = Path.cwd() / 'formatted_data'

#DOMAINS = ['attraction','hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train'] 
DOMAINS = ['hotel', 'restaurant', 'taxi', 'train']
#Special vocab tokens
PAD = '<pad>'
EOS = '</s>'
BOS = '<s>'
UNKNOWN = 'unk'

#Model specific special toks
DONTCARE = '<dontcare>'
NONE = '<none>'

SPECIAL_TOKS = [PAD, EOS, BOS, UNKNOWN, DONTCARE, NONE]

logger = logging.getLogger(__name__)

class Vocab():
    def __init__(self, n_grams, word_to_id=None):

        """
        --ngrams: (int) type of vocab, key in JSON repr of Vocab
        --word_to_id: (dict)
        """
        if word_to_id == None:
            word_to_id = {}
        self.word_to_id = word_to_id
        self.fill_special_toks()
        self.ngrams = n_grams
    
    def _input_to_indices(self, toks):
        """
        --toks: (List(str)) or (str) or (List(List(str)))
        """
        if type(toks) == str:
            return [[self.word_to_id[toks]]]
        elif type(toks[0]) == str:
            return [[self.word_to_id[tok] if tok in self.word_to_id.keys() else self.word_to_id[UNKNOWN] for tok in toks ]]
        elif type(toks[0]) == list:
            return [[self.word_to_id[tok] if tok in self.word_to_id.keys() else self.word_to_id[UNKNOWN] for tok in sent] for sent in toks]

    def to_idxs_tensor(self, input_to_embed, isDialogueVocab=False, device=None):
        """
        Converts a list of candidates/system dialogue acts to input tensor

        --input_to_embed: (str) or (List(str)) or (List(List(str)))
        --device: (torch.device)
        """
        indices = self._input_to_indices(input_to_embed)
        # Pad 
        # TODO: check with Josh about the purpose of the if-statement below
        if isDialogueVocab is False:
            indices = pad(indices, self.word_to_id[PAD])
        # print(indices)
        ind_tens = torch.tensor(indices, dtype=torch.int64, device=device)
        return ind_tens
        
    def new_from_domains(self, domains=DOMAINS, data_dir=DATA_DIR):
        """New Vocab from pickled formatted training data"""
        for d in domains:
            self.update_from_pkl(Path(data_dir) / '{}_dials_hyst_trainset.pkl'.format(d), 'pkl')

    def fill_special_toks(self):
        for i, sptok in enumerate(SPECIAL_TOKS):
            self.word_to_id[sptok] = i

    def update(self, tok, i):
        if tok not in self.word_to_id:
            self.word_to_id[tok] = i
            i += 1
        return i

    
    def update_from_pkl(self, pth, ftype): 
        """
        Reads all dials from pkl file and updates Vocab object
        
        --pth: Path object
        --ftype: one of {'json', 'pkl'}
        --usr: whether to get usr or system transcript
        l
        """
        i = len(self.word_to_id)
        if ftype == 'pkl':
            dials = pkl.load(open(pth, 'rb'))
        elif ftype == 'json':
            dials = json.load(open(pth, 'r'))
        for dial in dials:
            for turn in dial['dialogue'].keys():
                toks =  dial['dialogue'][turn]['user_utterance'] 
                candidates = dial['dialogue'][turn]['candidates']

                for t in toks + candidates:
                    i = self.update(t, i)
        
        # len of updated vocab
        return i  
    
    @classmethod
    def load_from_json(cls, pth, ngrams=None):
        """
            @ param ngrams (List[String]): list of characters which denotes what type of ngrams to include in the vocabulary (i.e ['1', '2'] --> unigram, bigram)
                                            ngrams set to None if we are loading in from the dialogue acts vocabulary

        """
        vocab_dict = {}
        if ngrams is not None:
            for ngram in ngrams:
                new_entries = json.load(open(pth, 'r'))[ngram]
                vocab_dict.update(new_entries)
                
        else:
            vocab_dict = json.load(open(pth, 'r'))

        return Vocab(ngrams, vocab_dict)

    def save_to_json(self, pth):
        pth = Path(pth)
        existing_vocab = {}
        if pth.exists():
            with open(pth, 'r') as f:
                existing_vocab = json.load(f)
                assert self.ngrams not in existing_vocab, "This file path already has a {}-gram vocab. Save to different file" \
                        .format(self.ngrams)
        existing_vocab.update({self.ngrams: self.word_to_id})
        with open(pth, 'w') as f:
            json.dump(existing_vocab, f, indent=2)
        logger.info("Saving {}-gram vocab to {}".format(self.ngrams, pth))
    
    def __len__(self):
        return len(self.word_to_id)

class DAVocab(Vocab):
    def __init__(self, word_to_id=None):
        if not word_to_id:
            word_to_id = {}
        self.word_to_id = word_to_id

    def update_from_pkl(self, pth, ftype):
        i = len(self.word_to_id)
        if ftype == 'pkl':
            dials = pkl.load(open(pth, 'rb'))
        elif ftype == 'json':
            dials = json.load(open(pth, 'r'))
        for dial in dials:
            for turn in dial['dialogue'].keys():
                acts = dial['dialogue'][turn].get('system_dialogue_acts', '')
                for act in acts:
                    i = self.update(act, i)
        return i

    def save_to_json(self, pth):
        pth = Path(pth)

        existing_vocab = {}
        if pth.exists():
            with open(pth, 'r') as f:
                existing_vocab = json.load(f)
        existing_vocab.update(self.word_to_id)

        with open(pth, 'w') as f:
            json.dump(existing_vocab, f, indent=2)
        logger.info('Saving Dialogue Act Vocab to json')



if __name__ == '__main__':
    """
    Run as script to build a Vocab from pickle file[s] and save to JSON

    """

    parser = ArgumentParser(description='Build an n_gram vocab')
    parser.add_argument('savepth', help='JSON path to save built Vocab too')
    parser.add_argument('--datadir', default=DATA_DIR, help='Optional data dir where pickled train data located, default is {}'.format(DATA_DIR))
    parser.add_argument('--ngrams', default=1, type=int, help='Type of Vocab')
    parser.add_argument('--domains', default=DOMAINS, nargs='+', help='Domains to build vocab from')
    parser.add_argument('--vocab', default='utterances', choices=['utterances', 'dialogue_acts'])
    args = parser.parse_args()
    if args.vocab == 'utterances':
        vocab = Vocab(args.ngrams)
        vocab.new_from_domains(args.domains, args.datadir)
        vocab.save_to_json(args.savepth)
    elif args.vocab == 'dialogue_acts':
        vocab = DAVocab()
        vocab.new_from_domains(args.domains, args.datadir)
        vocab.save_to_json(args.savepth)
