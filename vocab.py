import os
import nltk
import json
import pickle as pkl
import logging
from argparse import ArgumentParser
from pathlib import Path
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np

#Default dir 'formatted_data' in the parent dir of repo
DATA_DIR = Path.cwd() / 'formatted_data'

DOMAINS = ['attraction','hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train'] 

#Special vocab tokens
PAD = '<pad>'
EOS = '</s>'
BOS = '<s>'
UNKNOWN = 'unk'

logger = logging.getLogger(__name__)

class Vocab():
    def __init__(self, n_grams):

        """
        --ngrams: (int) type of vocab, key in JSON repr of Vocab

        """
        super().__init__()
        self.word_to_id = {}
        self.ngrams = n_grams

    def new_from_domains(self, domains, data_dir):
        """New Vocab from pickled formatted training data"""

        self.fill_special_toks()
        for d in domains:
            self.update_from_pkl(Path(data_dir) / '{}_dials_hyst.pkl'.format(d), 'pkl', self.ngrams)


    def fill_special_toks(self):
        self.word_to_id['<pad>'] = 0   # Pad Token
        self.word_to_id['<s>'] = 1 # Start Token
        self.word_to_id['</s>'] = 2    # End Token
        self.word_to_id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word_to_id['<unk>']

    def update(self, tok, i):
        if tok not in self.word_to_id:
            self.word_to_id[tok] = i
            i += 1
        return i

    
    def update_from_pkl(self, pth, ftype, usr=True):
        """
        Reads all dials from pkl file and updates Vocab object
        
        --pth: Path object
        --ftype: one of {'json', 'pkl'}
        --usr: whether to get usr or system transcript
        """
        trans_key = 'transcript' if usr else 'system_transcript'
        i = len(self.word_to_id)
        if ftype == 'pkl':
            dials = pkl.load(open(pth, 'rb'))
        elif ftype == 'json':
            dials = json.load(open(pth, 'r'))
        for dial in dials:
            for turn in dial['dialogue'].keys():
                toks = word_tokenize(dial['dialogue'][turn].get('user_utterance', ''))    
                if self.ngrams > 1:
                    toks = ngrams(toks, self.ngrams)
                    toks = [' '.join(tok) for tok in toks]
                for tok in toks:
                    i = self.update(tok, i)
        
        # len of updated vocab
        return i  
    
    def load_from_json(self, pth):
        return json.load(open(pth, 'r'))[self.ngrams]

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

if __name__ == '__main__':
    """
    Run as script to build a Vocab from pickle file[s] and save to JSON

    """

    parser = ArgumentParser(description='Build an n_gram vocab')
    parser.add_argument('savepth', help='JSON path to save built Vocab too')
    parser.add_argument('--datadir', default=DATA_DIR, help='Optional data dir where pickled train data located, default is {}'.format(DATA_DIR))
    parser.add_argument('--ngrams', default=1, type=int, help='Type of Vocab')
    parser.add_argument('--domains', default=DOMAINS, nargs='+', help='Domains to build vocab from')
    args = parser.parse_args()
    vocab = Vocab(args.ngrams)
    vocab.new_from_domains(args.domains, args.datadir)
    vocab.save_to_json(args.savepth)

