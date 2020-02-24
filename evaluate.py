""" evaluate.py: evaluates the model """

import argparse
import logging
import os

import numpy as np 
import torch
import utils

from model.DSTModel import DST
from model.DSTModel import goal_accuracy_metric
from model.data_loader import DialoguesDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import torch.nn.functional as F
import torch.nn as nn



def evaluate(model, evaluation_data, model_dir, dataset_params, device):
    """ Evaluates the model over the evaluation data """

    batch_size = dataset_params['batch_size']
    num_of_slots = 35

    # set model in evaluation model
    model.eval()

    # set up validation_generator --> data iterator wich generates batches for the entire dataset
    validation_generator = evaluation_data.data_iterator(batch_size=dataset_params['batch_size'], shuffle=False) 

    total_loss_eval = 0

    num_of_steps = evaluation_data.__len__() // batch_size

    # summary for current eval loop
    summ = []

    t = trange(num_of_steps)

    for i in t:
        try:
            turn_and_cand, cand_label = next(validation_generator)
            context_vectors = torch.stack([model.get_turncontext(cand_dict) for cand_dict in turn_and_cand])
            candidates = [cand_dict['candidate'] for cand_dict in turn_and_cand]

            output = model.forward(context_vectors, candidates) 
            output = output.squeeze(dim=1)

            # need to weight loss due to the imbalance in positive to negative examples 
            weights = [300.0] * num_of_slots
            pos_weights = torch.tensor(weights)
            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
            loss = loss_func(output, cand_label)

            # generate summary statistics
            accuracy = goal_accuracy_metric(output, cand_label).item()
            total_loss = loss.sum().item()
            avg_loss_batch = total_loss/(batch_size * num_of_slots)

            summary_batch = {'goal_accuracy' : accuracy,
                            'total_loss' : total_loss,
                            'avg_loss' : avg_loss_batch
                    }

            summ.append(summary_batch)
            
            # add to total loss
            total_loss_eval += total_loss

        # no more batches left
        except StopIteration:
            break

        avg_loss_eval = total_loss_eval/((i+1) * batch_size * num_of_slots)

    
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    logging.info("Average Evaluation Loss: {}".format(avg_loss_eval))     

    return metrics_mean   

