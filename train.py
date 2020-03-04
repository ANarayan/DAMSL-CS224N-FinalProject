import argparse
from torch.autograd import Variable
import logging
import json
import os
import random
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np 
import torch.optim as optim
from tqdm import trange

import utils 
from model.DSTModel import DST
from model.data_loader import DialoguesDataset
from model.DSTModel import goal_accuracy_metric
from evaluate import evaluate
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', default='data/', help='Directory which contain the dialogue dataset')
parser.add_argument('--model_dir', default='experiments/', help="Directory containing model config file")

def train(model, training_data, optimizer, model_dir, training_params, dataset_params, device):
    """ Trains over the entire training data set """

    model.train()

    batch_size = dataset_params['batch_size']
    training_generator = training_data.data_iterator(batch_size=dataset_params['batch_size'], shuffle=False)
    validation_generator = validation_data.data_iterator(batch_size=dataset_params['batch_size'], shuffle=False)

    summ = []
    num_of_slots = 35
    total_loss_train = 0
    num_steps = training_data.__len__() // batch_size
    pos_weights = torch.tensor([training_params['pos_weighting']] * num_of_slots)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
        
    t = trange(num_steps)

    for i in t:
        try:
            turn_and_cand, cand_label = next(training_generator)
            output = model(turn_and_cand)
            # need to weight loss due to the imbalance in positive to negative examples 
            # Confirm the 300
            loss = loss_func(output, cand_label) # Tensor: (batch_size, #_of_slots=35)


            # clear prev. gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.sum().backward()

            # perform update
            optimizer.step()

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
            total_loss_train += loss.sum().item()

        # no more batches left
        except StopIteration:
            break
    
        avg_loss_train = total_loss_train/((i+1) * batch_size * num_of_slots)
    
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics : " + metrics_string)
    logging.info("Average Training loss: {}".format(avg_loss_train))


def train_and_eval(model, training_data, validation_data, optimizer, model_dir, training_params, dataset_params, device):
    """ Trains and evaluates the model for the specified number of epochs """
    total_epochs = training_params['num_epochs']
    best_val_acc = 0.0

    for epoch in range(total_epochs):
        logging.info("Epoch {}/{}".format(epoch+1,total_epochs))
        
        # Train model
        train(model, training_data, optimizer, model_dir, training_params, dataset_params, device)

        # Evaluate model
        eval_metrics = evaluate(model, validation_data, model_dir, dataset_params, device)

        val_acc = eval_metrics['goal_accuracy']
        is_best_acc = best_val_acc <= val_acc

        # Save model 
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                                checkpoint=model_dir,
                                is_best=is_best_acc)

        # If best_eval, best_save_path
        if is_best_acc:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.pkl")
            utils.save_dict_to_pkl(eval_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.pkl")
        utils.save_dict_to_pkl(eval_metrics, last_json_path)



if __name__ == '__main__':

    USING_MODEL_CONFIGFILE = False
    #TRAIN_FILE_NAME = 'restaurant_hyst_train.pkl'
    TRAIN_FILE_NAME = 'single_pt_dataset.pkl'
    #VAL_FILE_NAME = 'restaurant_hyst_val.pkl'
    VAL_FILE_NAME = 'single_pt_dataset.pkl'
    TEST_FILE_NAME = 'restaurant_hyst_test.pkl'


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # first load parameters from params.json
    args = parser.parse_args()

    # Load in candidate vocab
    with open('vocab.json') as cand_vocab:
        candidate_vocabulary = json.load(cand_vocab)['1']

    if USING_MODEL_CONFIGFILE:
        json_path = os.path.join(args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json config file gound at {}".format(json_path)
        params = utils.read_json_file(json_path)
        model_params = params['model']
        training_params = params['training']
        dataset_params = params['data']
    else:
        model_params = {
            'embed_dim' : 300, 
            'sentence_hidden_dim' : 256, 
            'hierarchial_hidden_dim' : 512,
            'da_hidden_dim' : 64, 
            'da_embed_size' : 50,
            'ff_hidden_dim' : 256, 
            'batch_size' : 2,
            'num_slots' : 35,
            'ngrams' : '1',
            'candidate_utterance_vocab_pth' : 'vocab.json',
            'da_vocab_pth': 'davocab.json'
        }

        training_params = {
            'num_epochs' : 10,
            'learning_rate' : 0.001,
            'pos_weighting' : 20
        }


        dataset_params = {
            'batch_size': 2,
            'shuffle': True,
            'num_workers': 1
        }

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading the datasets...")

    # Load data
    training_data = DialoguesDataset(os.path.join(args.data_dir, TRAIN_FILE_NAME))
    validation_data = DialoguesDataset(os.path.join(args.data_dir, VAL_FILE_NAME))

    logging.info("-done")

    # define model and optimizer
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params)

    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(training_params['num_epochs']))

    train_and_eval(model, training_data, validation_data, optimizer, args.model_dir, training_params, dataset_params, device)
