import argparse
from torch.autograd import Variable
import logging
import json
import os

import torch
import numpy as np 
import torch.optim as optim
from tqdm import trange

import utils 
from model.DSTModel import DST
from model.data_loader import DialoguesDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', default='data/', help='Directory which contain the dialogue dataset')
parser.add_argument('--model_dir', default='experiments/', help="Directory containing model config file")

def train(model, training_data, validation_data, optimizer, model_dir, training_params, dataset_params):
    model.train()
    #training_generator = DataLoader(training_data, **dataset_params)
    #validation_generator = DataLoader(validation_data, **dataset_params)

    training_generator = training_data.data_iterator(3, False)
    validation_generator = validation_data.data_iterator(3, False)

    total_epochs = training_params['num_epochs']

    for epoch in range(total_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, total_epochs))

        total_loss_train = 0
        total_loss_eval = 0

        # TRAINING
        for i in range(training_data.__len__()):
            try:
                turn_and_cand, cand_label = next(training_generator)

                context_vectors = torch.stack([model.get_turncontext(cand_dict) for cand_dict in turn_and_cand])
                candidates = torch.tensor([cand_dict['candidate'] for cand_dict in turn_and_cand])

                output = model.forward(context_vectors, candidates) 

                # need to weight loss due to the imbalance in positive to negative examples 
                loss_func = nn.BCELoss(weight= 20, reduction='none')
                loss = loss_func(output, cand_label)

                # clear prev. gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # perform update
                optimizer.step()
            
                # add to total loss
                total_loss_train += loss.item()

            # no more batches left
            except StopIteration:
                break
        
        avg_loss_train = total_loss_train/len(training_generator)
        logging.info("Average Training loss: {}".format(avg_loss_train))


        # Evaluation
        for i in range(training_data.__len__()):
            try:
                turn_and_cand, cand_label = next(training_generator)

                context_vectors = torch.stack([model.get_turncontext(cand_dict) for cand_dict in turn_and_cand])
                candidates = torch.tensor([cand_dict['candidate'] for cand_dict in turn_and_cand])

                output = model.forward(context_vectors, candidates) 

                # need to weight loss due to the imbalance in positive to negative examples 
                loss_func = nn.BCELoss(weight= 20, reduction='none')
                loss = loss_func(output, cand_label)

                # clear prev. gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # perform update
                optimizer.step()
            
                # add to total loss
                total_loss_eval += loss.item()

            # no more batches left
            except StopIteration:
                break

        
        avg_loss_eval = total_loss_eval/len(validation_generator)
        logging.info("Average Training loss: {}".format(avg_loss_eval))
        

if __name__ == '__main__':

    USING_MODEL_CONFIGFILE = False
    TRAIN_FILE_NAME = 'restaurant_hyst_train.pkl'
    VAL_FILE_NAME = 'restaurant_hyst_val.pkl'
    TEST_FILE_NAME = 'restaurant_hyst_test.pkl'

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
            'da_hidden_dim' : 16, 
            'da_embed_size' : 64,
            'ff_hidden_dim' : 256, 
            'batch_size' : 32,
            'num_slots' : 35,
            'vocab' : candidate_vocabulary
        }

        training_params = {
            'num_epochs' : 5,
            'learning_rate' : 0.001
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

    train(model, training_data, validation_data, optimizer, args.model_dir, training_params, dataset_params)
