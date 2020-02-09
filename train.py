import argparse
import logging
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


parser.add_argument('--data_dir', help='Directory which contain the dialogue dataset')
parser.add_argument('--model_dir', help="Directory containing model config file")

def train(model, training_data, validation_data, optimizer, model_dir, training_params, dataset_params):
    model.train()
    training_generator = DataLoader(training_data, **dataset_params)
    validation_generator = DataLoader(validation_data, **dataset_params)

    total_epochs = training_params['num_epochs']

    for epoch in range(total_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, total_epochs))

        total_loss_train = 0
        total_loss_eval = 0

        # TRAINING
        for turn, turn_label in training_generator:
            # Generate context feature vector
            # Compute loss for each candidate
            context_vector = model.get_turncontext(turn)
            
            candidates = turn['candidates']
            # iterate through each candidate and compute the loss
            for index, candidate in enumerate(candidates):
                # candidate_gtlabel --> Dim: (# of slots x 1)
                candidate_gtlabel = turn_label[index].unsqueeze(1)

                # compute model output + loss on the turn + candidate pair
                output = model.forward(context_vector, candidate) 
                loss_func = nn.BCELoss(reduction='none')
                loss = loss_func(output, candidate_gtlabel)

                # clear prev. gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # perform update
                optimizer.step()
            
                # add to total loss
                total_loss_train += loss.item()
        
        avg_loss_train = total_loss_train/len(training_generator)
        logging.info("Average Training loss: {}".format(avg_loss_train))

        # EVALUATION
        for turn, turn_label in validation_generator:
            # Generate context feature vector
            # Compute loss for each candidate
            context_vector = model.get_turncontext(turn)
            
            candidates = turn['candidates']
            # iterate through each candidate and compute the loss
            for index, candidate in enumerate(candidates):
                # candidate_gtlabel --> Dim: (# of slots x 1)
                candidate_gtlabel = turn_label[index].unsqueeze(1)

                # compute model output + loss on the turn + candidate pair
                output = model.forward(context_vector, candidate) 
                loss_func = nn.BCELoss(reduction='none')
                loss = loss_func(output, candidate_gtlabel)

                # clear prev. gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # perform update
                optimizer.step()
            
                # add to total loss
                total_loss_eval += loss.item()
        
        avg_loss_eval = total_loss_eval/len(validation_generator)
        logging.info("Average Training loss: {}".format(avg_loss_eval))
        

if __name__ == '__main__':

    USING_MODEL_CONFIGFILE = False
    # first load parameters from params.json
    args = parser.parse_args()

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
        }

        training_params = {
            'num_epochs' : 5,
            'learning_rate' : 0.001
        }

        dataset_params = {
            'batch_size': 1,
            'shuffle': True,
            'num_workers': 1
        }


    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading the datasets...")

    # Load data
    training_data = DialoguesDataset(os.path.join(args.data_dir, '/train.pkl'))
    validation_data = DialoguesDataset(os.path.join(args.data_dir, '/val.pkl'))
    
    logging.info("-done")

    # define model and optimizer
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params).cuda()

    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(training_params['num_epochs']))

    train(model, training_data, validation_data, optimizer, args.model_dir, training_params, dataset_params)
