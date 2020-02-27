import os
import torch
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pickle
import random


import warnings 
warnings.filterwarnings("ignore")

class DialoguesDataset(Dataset):
    """ Dialogues dataset """
    def __init__(self, data_file):
        """
        Args:
            data_file (string): Path to pickle file with turn+candidate pairs.
        """
        dialogue_data = open(data_file, 'rb') 
        self.turn_cand_dps= pickle.load(dialogue_data)[0:284800] # need to make sure it divisible by all batch sizes
     
    def __len__(self):
        return len(self.turn_cand_dps)

    def __getitem__(self, turn_idx):
        """
            @param dialogue_idx (int): idx of dialogue we are querying data from
            @return turn (Dict): a dictionary which contains context for a randomly 
                                sampled invidual turn in the given dialogue
        """
        """
        turn_cand_dict : {
            'user_utterance': List[String],
            'utterance_history': List[List[String]],
            'system_dialogue_acts': List[String],
            'candidate' : String,
            'gt_label' : array
        }
        """
        turn_cand_dict = self.turn_cand_dps[turn_idx]

        # get label and convert to tensor
        label = torch.Tensor(turn_cand_dict['gt_label'])

        # returns turn information and gt labels associated with each candidate for in the turn
        return turn_cand_dict

    def data_iterator(self, batch_size=1, shuffle=False):
        # print(batch_size)
        order = list(range(self.__len__()))
        if shuffle:
            random.seed(230)
            random.shuffle(order)
        
        # take one pass over data in the dataset
        for i in range(self.__len__() // batch_size):
            batch_datapoints = [self.__getitem__(idx) for idx in order[i*batch_size: (i+1)*batch_size]]
            batch_labels = [torch.Tensor(datapoint['gt_label']) for datapoint in batch_datapoints]

            batch_data, batch_labels = batch_datapoints, torch.stack(batch_labels)

            if torch.cuda.is_available():
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
            
            #print(batch_data.type)

            yield batch_data, batch_labels






    



        


