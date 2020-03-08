import argparse
from torch.autograd import Variable
import logging
import json
import os
import random
from pathlib import Path
from copy import deepcopy

import torch
import torchvision
import numpy as np 
import torch.optim as optim
from tqdm import trange

import utils 
from model.DSTModel import DST
from model.data_loader import DialoguesDataset
from evaluate import evaluate
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', default='data/', help='Directory which contain the dialogue dataset')
parser.add_argument('--model_dir', default='experiments/', help="Directory containing model config file")
parser.add_argument('--output_model_dir', default='experiments/', help="Directory for saving training logs + model outputs")
parser.add_argument('--fine_tune_domain', default=None, help='Domain to finetune on')

def train(model, training_data, optimizer, model_dir, training_params, dataset_params, device):
    """ Trains over the entire training data set """

    model.train()

    batch_size = dataset_params['train_batch_size']
    training_generator = training_data.data_iterator(batch_size=dataset_params['train_batch_size'], shuffle=True, is_train=True)

    summ = []
    total_loss_train = 0

    num_steps = training_data.__len__() // batch_size
        
    t = trange(num_steps)

    for i in t:
        try:
            turn_and_cand, cand_label = next(training_generator)
            output = model.forward(turn_and_cand) # Tensor: (batch_size, 1, embed_size)
            output = output.squeeze(dim=1).cpu()

            # need to weight loss due to the imbalance in positive to negative examples 
            if training_params['pos_weighting'] is not None:
                pos_weights = training_params['pos_weighting']
            else:
                pos_weights = torch.tensor([1.0] * training_params['num_slots'])

            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
            loss = loss_func(output, cand_label) # Tensor: (batch_size, #_of_slots=35)

            # clear prev. gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.sum().backward()

            # perform update
            optimizer.step()

            batch_loss = loss.sum().item()

            summary_batch = {
                            'batch_loss' : batch_loss,
                    }
            summ.append(summary_batch)
            
            # add to total loss
            total_loss_train += batch_loss

        # no more batches left
        except StopIteration:
            break

    
    avg_loss_train = total_loss_train/(num_steps)
 
    
    metrics_mean = {metric:np.mean([x[metric] for x in summ if x[metric] is not None]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics : " + metrics_string)
    logging.info("Average Training loss: {}".format(avg_loss_train))

    return total_loss_train, avg_loss_train


def train_and_eval(model, training_data, validation_data, optimizer, model_dir, training_params, dataset_params, device):
    """ Trains and evaluates the model for the specified number of epochs """
    total_epochs = training_params['num_epochs']
    best_val_acc = 0.0
    prev_val_acc = 0.0
    best_loss = 1e1000
    prev_val_loss = 1e1000
    early_stopping_count = 0

    for epoch in range(total_epochs):
        logging.info("Epoch {}/{}".format(epoch+1,total_epochs))
        
        # Train model
        total_loss_train, avg_loss_train = train(model, training_data, optimizer, model_dir, training_params, dataset_params, device)

        # Evaluate model
        eval_metrics, total_loss_eval, eval_avg_goal_acc, eval_joint_goal_acc, avg_slot_precision = evaluate(model, validation_data, model_dir, dataset_params, device)

        """
        Accuracy to do HP tuning:
            val_acc = eval_avg_goal_acc
            is_best_acc = val_acc >= best_val_acc
        """

        #Using loss to do HP tuning
        val_loss = total_loss_eval
        is_best_loss = val_loss <= best_loss

        # Save model 
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'loss' : total_loss_eval,
                               'avg. goal accuracy' : eval_avg_goal_acc,
                               },
                                checkpoint=model_dir,
                                is_best=is_best_loss)

        # If best_eval, best_save_path
        if is_best_loss:
            logging.info("- Found new best loss")
            best_loss = val_loss

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.pkl")
            utils.save_dict_to_pkl(eval_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.pkl")
        utils.save_dict_to_pkl(eval_metrics, last_json_path)

        # Plot train + eval loss
        eval_writer.add_scalar('loss', total_loss_eval, epoch)
        train_writer.add_scalar('loss', total_loss_train, epoch)

        # plot avg. slot accuracy, joint goal accuracy and slot precision
        eval_writer.add_scalar('average_goal_accuracy', eval_avg_goal_acc, epoch)
        eval_writer.add_scalar('joint_goal_accuracy', eval_joint_goal_acc, epoch)
        eval_writer.add_scalar('avg_slot_precision', avg_slot_precision, epoch)


        # Early stopping --> loss increases for greater than 2 epochs, stop
        if val_loss >= prev_val_loss:
            early_stopping_count += 1
            if early_stopping_count >=2:
                break

        prev_val_loss = val_loss

    return best_loss

if __name__ == '__main__':

    USING_MODEL_CONFIGFILE = False
    MODEL_CHECKPOINT = True

    TRAIN_FILE_NAME = 'mst_attraction_train.pkl'
    #TRAIN_FILE_NAME = 'single_pt_dataset.pkl'
    VAL_FILE_NAME = 'mst_attraction_val.pkl'
    #VAL_FILE_NAME = 'single_pt_dataset.pkl'
    TEST_FILE_NAME = 'mst_attraction_test.pkl'

    # first load parameters from params.json
    args = parser.parse_args()

    experiment_name = args.model_dir.split('/')[-1]
    train_writer = SummaryWriter(comment = "_train" + experiment_name )
    eval_writer = SummaryWriter(comment = "_eval" + experiment_name )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params_json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_json_path), "No json config file found at {}".format(params_json_path)
    params = utils.read_json_file(params_json_path)

    # get positive weightage
    if args.fine_tune_domain is not None:
        weightage_path = os.path.join(args.model_dir, 'domain_pos_weights.json')
        assert os.path.isfile(weightage_path), "No json config file found at {}".format(weightage_path)
        weights_dict = utils.read_json_file(weightage_path)
        pos_weights = weights_dict[args.fine_tune_domain]
    else:
        pos_weights = None
    
    
    model_params = {
        'embed_dim' : 300, 
        'sentence_hidden_dim' : params['sentence_hidden_dim'], 
        'hierarchial_hidden_dim' : params['hierarchial_hidden_dim'],
        'da_hidden_dim' : params[ 'da_hidden_dim'], 
        'da_embed_size' : 50,
        'ff_hidden_dim' : params['ff_hidden_dim'], 
        'ff_dropout_prob' : params[ 'ff_dropout_prob'],
        'batch_size' : params['batch_size'],
        'num_slots' : 35,
        'ngrams' : ['3'],
        'canzdidate_utterance_vocab_pth' : 'mst_attraction_vocab.json',
        'da_vocab_pth': 'mst_attraction_davocab.json',
        'device' : device
    }

    training_params = {
        'num_epochs' : 20,
        'learning_rate' : params['learning_rate'],
        'pos_weighting' : pos_weights,
        'train_set_percentage' : 1, # used for fine-tuning experiments
    }

    dataset_params = {
        'train_batch_size': params['batch_size'],
        'eval_batch_size' : 1,
        'shuffle': True,
        'num_workers': 1,
        'num_of_slots' : 35
    }

    utils.set_logger(os.path.join(args.ouput_model_dir, 'train.log'))
    logging.info("Loading the datasets...")

    # Load data
    training_data = DialoguesDataset(os.path.join(args.data_dir, TRAIN_FILE_NAME), training_params['train_set_percentage'])
    validation_data = DialoguesDataset(os.path.join(args.data_dir, VAL_FILE_NAME))
    test_data = DialoguesDataset(os.path.join(args.data_dir, TEST_FILE_NAME))

    logging.info("-done")

    # define model and optimizer
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params)

    if MODEL_CHECKPOINT:
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
    
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(training_params['num_epochs']))

    best_val_loss = train_and_eval(model, training_data, validation_data, optimizer, args.model_dir, training_params, dataset_params, device)
    
    """
    # TODO: Need to update which of the current parameter configurations yielded the best results! Read in file, compare best va;
    # acc with current accuracy and replace or dont replace.
    

    # Update which model config yielded the best results
    json_path = os.path.join('experiments/', 'best_config.json')
    best_results_dict = utils.read_json_file(json_path)
    curr_best_loss = best_results_dict['best_val_loss']

    if best_val_loss < curr_best_loss:
        best_results_dict['best_val_loss'] = best_val_loss
        best_results_dict['experiment_name'] = experiment_name
        utils.write_json_file(best_results_dict, json_path)
    
    """
    test_model = DST(**model_params).cuda()
    utils.load_checkpoint(os.path.join(args.output_model_dir, 'best.pth.tar'), test_model)
    # Run on test data
    logging.info("TEST SET METRICS ----------------  : ")
    eval_metrics, total_loss_eval, eval_avg_goal_acc, eval_joint_goal_acc, avg_slot_precision = evaluate(test_model, test_data, args.output_model_dir, dataset_params, device)
    train_writer.close()
    eval_writer.close()
