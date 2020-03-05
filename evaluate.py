""" evaluate.py: evaluates the model """

import argparse
import logging
import os

import numpy as np 
import torch
import utils

from model.DSTModel import DST
from model.DSTModel import goal_accuracy
from model.data_loader import DialoguesDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    total_tps, total_fps,total_tns, total_fns, correct_class, total_class = 0, 0, 0, 0, 0,0
    joint_goal_acc_sum = 0
    avg_goal_acc_sum = 0

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
            output = output.squeeze(dim=1).cpu()

            # need to weight loss due to the imbalance in positive to negative examples 
            weights = [20.0] * num_of_slots
            pos_weights = torch.tensor(weights)
            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
            loss = loss_func(output, cand_label)


            # generate summary statistics
            true_pos, false_pos, true_neg, false_neg, joint_goal_acc, goal_acc, all_matches, all_class = goal_accuracy(output, cand_label)
            # goal accuracy a.k.a precision
            total_tps += true_pos
            total_fps += false_pos
            total_tns += true_neg
            total_fns += false_neg
            joint_goal_acc_sum += joint_goal_acc
            avg_goal_acc_sum += goal_acc

            correct_class += all_matches
            total_class += all_class

            batch_loss = loss.sum().item()

            summary_batch = {
                            'batch_loss' : batch_loss,
                        }
            summ.append(summary_batch)
    
           
            # add to total loss
            total_loss_eval += batch_loss

        # no more batches left
        except StopIteration:
            break

    avg_loss_eval = total_loss_eval/(num_of_steps)
    joint_goal_acc = joint_goal_acc_sum/(num_of_steps)
    avg_goal_acc  = correct_class/total_class

    precision = total_tps/(total_tps + total_fps) if (total_tps + total_fps) != 0 else 0 
    recall = (total_tps + total_tns)/(total_tps + total_fns + total_tns + total_fps) if (total_tps + total_fns) != 0 else 0 
    f1 = 2 * (precision * recall)/(precision + recall) if precision != 0 and recall != 0 else 0

    
    metrics_mean = {metric:np.mean([x[metric] for x in summ if x[metric] is not None]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    logging.info("Average Evaluation Loss: {}".format(avg_loss_eval))  
    logging.info("Eval Precision: {}; Recall: {}; F1: {}".format(precision, recall, f1))  
    logging.info("Joint goal accuracy: {}".format(joint_goal_acc))
    logging.info("Average goal accuracy: {}".format(avg_goal_acc))

    metrics_mean['avg_goal_accuracy'] = avg_goal_acc
    metrics_mean['joint_goal_accuracy'] = joint_goal_acc

    return metrics_mean, total_loss_eval, avg_goal_acc, joint_goal_acc

