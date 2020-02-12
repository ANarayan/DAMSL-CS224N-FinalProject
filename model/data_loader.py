import os
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import pickle
import random


import warnings 
warnings.filterwarnings("ignore")

class DialoguesDataset(Dataset):
    """ Dialogues dataset """
    def __init__(self, data_file):
        """
        Args:
            data_file (string): Path to pickle file with dialogues.
        """
        dialogue_data = open(data_file, 'rb') 
        self.dialogues = pickle.load(dialogue_data)
    
    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, dialogue_idx):
        """
            @param dialogue_idx (int): idx of dialogue we are querying data from
            @return turn (Dict): a dictionary which contains context for a randomly 
                                sampled invidual turn in the given dialogue
        """
        dialogue_dict = self.dialogues[dialogue_idx]
        dialogue_turns = dialogue_dict['dialogue']
        turn_keys = list(dialogue_turns.keys())
        random_turn_idx = random.choice(turn_keys)

        turn_dict = dialogue_turns[random_turn_idx]
        turn_dict['utterance_history'] = []

        for idx in range(0, random_turn_idx):
            turn_dict['utterance_history'].append(dialogue_turns[idx]['user_utterance'])
        
        # add the current turn utterances to the history
        turn_dict['utterance_history'].append(turn_dict['user_utterance'])

        # get label and convert to tensor
        label = torch.Tensor(turn_dict['gt_labels'])

        # returns turn information and gt labels associated with each candidate for in the turn
        return turn_dict, label

