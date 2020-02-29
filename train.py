import argparse
from torch.autograd import Variable
import logging
import json
import os

import torch
import torchvision
import numpy as np 
import torch.optim as optim
from tqdm import trange

import utils 
from model.DSTModel import DST
from model.data_loader import DialoguesDataset
from model.DSTModel import goal_accuracy
from evaluate import evaluate
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', default='data/', help='Directory which contain the dialogue dataset')
parser.add_argument('--model_dir', default='experiments/', help="Directory containing model config file")

def train(model, training_data, optimizer, model_dir, training_params, dataset_params, device):
    """ Trains over the entire training data set """

    model.train()

    batch_size = dataset_params['batch_size']
    training_generator = training_data.data_iterator(batch_size=dataset_params['batch_size'], shuffle=True)

    summ = []
    num_of_slots = 35
    total_loss_train = 0
    total_tps, total_fps, total_fns = 0, 0, 0 
    joint_goal_acc_sum = 0
    avg_goal_acc_sum = 0

    num_steps = training_data.__len__() // batch_size
        
    t = trange(num_steps)

    for i in t:
        try:
            turn_and_cand, cand_label = next(training_generator)
            context_vectors = torch.stack([model.get_turncontext(cand_dict) for cand_dict in turn_and_cand])
            candidates = [cand_dict['candidate'] for cand_dict in turn_and_cand]
            output = model.forward(context_vectors, candidates) # Tensor: (batch_size, 1, embed_size)
            output = output.squeeze(dim=1)

            # need to weight loss due to the imbalance in positive to negative examples 
            # Confirm the 300
            weights = [20.0] * num_of_slots
            pos_weights = torch.tensor(weights)
            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
            loss = loss_func(output, cand_label) # Tensor: (batch_size, #_of_slots=35)


            # clear prev. gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.sum().backward()

            # perform update
            optimizer.step()

            # generate summary statistics
            true_pos, false_pos, false_neg, joint_goal_acc, goal_acc = goal_accuracy(output, cand_label)
            # goal accuracy a.k.a precision
            total_tps += true_pos
            total_fps += false_pos
            total_fns += false_neg
            joint_goal_acc_sum += joint_goal_acc
            avg_goal_acc_sum += goal_acc

            batch_loss = loss.sum().item()

            summary_batch = {
                            'total_batch_loss' : batch_loss,
                    }
            summ.append(summary_batch)
            
            # add to total loss
            total_loss_train += batch_loss

        # no more batches left
        except StopIteration:
            break

    
    avg_loss_train = total_loss_train/(num_steps)
    joint_goal_acc = joint_goal_acc_sum/(num_steps)
    avg_goal_acc  = (total_tps)/(total_tps + total_fns)

    precision = total_tps/(total_tps + total_fps) if (total_tps + total_fps) != 0 else 0 
    recall = total_tps/(total_tps + total_fns) if (total_tps + total_fns) != 0 else 0 
    f1 = 2 * (precision * recall)/(precision + recall) if precision != 0 and recall != 0 else 0
    
    metrics_mean = {metric:np.mean([x[metric] for x in summ if x[metric] is not None]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics : " + metrics_string)
    logging.info("Average Training loss: {}".format(avg_loss_train))
    logging.info("Train Precision: {}; Recall: {}; F1: {}".format(precision, recall, f1)) 
    logging.info("Joint goal accuracy: {}".format(joint_goal_acc))
    logging.info("Average goal accuracy: {}".format(avg_goal_acc)) 
    return total_loss_train, avg_goal_acc, joint_goal_acc


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
        total_loss_train, train_avg_goal_acc, train_joint_goal_acc = train(model, training_data, optimizer, model_dir, training_params, dataset_params, device)

        # Evaluate model
        eval_metrics, total_loss_eval, eval_avg_goal_acc, eval_joint_goal_acc = evaluate(model, validation_data, model_dir, dataset_params, device)

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

        # plot avg. slot accuracy
        eval_writer.add_scalar('average_goal_accuracy', train_avg_goal_acc, epoch)
        train_writer.add_scalar('average_goal_accuracy', eval_avg_goal_acc, epoch)

        """# plot joint slot accuracy
        eval_writer.add_scalar('joint_goal_accuracy', train_joint_goal_acc, epoch)
        train_writer.add_scalar('joint_goal_accuracy', eval_joint_goal_acc, epoch)"""

        # TODO: Need to implement early stopping: if acc. decreases for two consecutive epochs, stop training
        if val_loss >= prev_val_loss:
            early_stopping_count += 1
            if early_stopping_count >= 3:
                break

        prev_val_loss = val_loss

    return best_loss

if __name__ == '__main__':

    USING_MODEL_CONFIGFILE = False
    TRAIN_FILE_NAME = 'attraction_hyst_train.pkl'
    #TRAIN_FILE_NAME = 'single_pt_dataset.pkl'
    VAL_FILE_NAME = 'attraction_hyst_val.pkl'
    #VAL_FILE_NAME = 'single_pt_dataset.pkl'
    TEST_FILE_NAME = 'restaurant_hyst_test.pkl'

    # first load parameters from params.json
    args = parser.parse_args()

    experiment_name = args.model_dir.split('/')[-1]
    train_writer = SummaryWriter(comment = "_train" + experiment_name )
    eval_writer = SummaryWriter(comment = "_eval" + experiment_name )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load in candidate vocab
    with open('vocab.json') as cand_vocab:
        candidate_vocabulary = json.load(cand_vocab)['1']

    json_path = os.path.join('experiments/', 'params.json')
    assert os.path.isfile(json_path), "No json config file gound at {}".format(json_path)
    params = utils.read_json_file(json_path)
        

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
        'ngrams' : ['1', '2'],
        'candidate_utterance_vocab_pth' : 'vocab.json',
        'da_vocab_pth': 'davocab.json',
        'device' : device
    }

    training_params = {
        'num_epochs' : 20,
        'learning_rate' : params['learning_rate']
    }

    dataset_params = {
        'batch_size': params['batch_size'],
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

    best_val_loss = train_and_eval(model, training_data, validation_data, optimizer, args.model_dir, training_params, dataset_params, device)
    # TODO: Need to update which of the current parameter configurations yielded the best results! Read in file, compare best va;
    # acc with current accuracy and replace or dont replace.

    # Update which model config yielded the best results
    json_path = os.path.join('experiments/', 'best_config.json')
    best_results_dict = utils.read_json_file(json_path)
    curr_best_loss = best_results_dict['best_val_loss']

    if best_val_loss < curr_best_loss:
        best_results_dict['best_acc'] = best_val_loss
        best_results_dict['experiment_name'] = experiment_name
        utils.write_json_file(best_results_dict, json_path)
    

    train_writer.close()
    eval_writer.close()
