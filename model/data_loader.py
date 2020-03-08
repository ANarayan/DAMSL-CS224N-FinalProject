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
    def __init__(self, data_file, dataset_percentage=1):
        """
        Args:
            data_file (string): Path to pickle file with turn+candidate pairs.
        """
        dialogue_data = open(data_file, 'rb') 
        
        # Randomly shuffle dataset
        random.seed(30)
        random.shuffle(dialogue_data)

        # For the purposes of testing (i.e. fine-tuning), use only a subset of datset examples
        dataset_split_idx = dataset_percentage * len(dialogue_data)
        dialogue_data = dialogue_data[:dataset_split_idx] 

        self.turn_cand_dps= pickle.load(dialogue_data) # need to make sure it divisible by all batch sizes
     
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

        """# get label and convert to tensor
        if is_train:
            label = torch.Tensor(turn_cand_dict['gt_label'])
        else:"""

        # returns turn information and gt labels associated with each candidate for in the turn
        return turn_cand_dict

    def data_iterator(self, batch_size=1, shuffle=False, is_train=True):
        # print(batch_size)
        order = list(range(self.__len__()))
        if shuffle:
            random.seed(230)
            random.shuffle(order)
        
        # if is_train, data points are in the format (cand, turn context)
        if is_train:
            # take one pass over data in the dataset
            for i in range(self.__len__() // batch_size):
                batch_datapoints = [self.__getitem__(idx) for idx in order[i*batch_size: (i+1)*batch_size]]
                batch_labels = [torch.Tensor(datapoint['gt_label']) for datapoint in batch_datapoints]

                batch_data, batch_labels = batch_datapoints, torch.stack(batch_labels)

                yield batch_data, batch_labels

        # else, if it is the validation and test set, each data point is one entire turn
        else:
            # Note: batch_size = 1 for validation and test
            # take one pass over data in the dataset
            for i in range(self.__len__() // batch_size):
                batch_turns = [self.__getitem__(idx) for idx in order[i*batch_size: (i+1)*batch_size]]
                batch_labels = [torch.Tensor(turn['gt_labels']) for turn in batch_turns]

                # batch_data: turn context, batch_labels: gt_annotation for each candidate in the turn
                batch_data, batch_labels = batch_turns[0], batch_labels[0]

                yield batch_data, batch_labels


#TODO: if time, write a batch loader for tasks/domains, to make maml extensible
# to many domains by not having to load all their data into memory simultaneously




    



        


