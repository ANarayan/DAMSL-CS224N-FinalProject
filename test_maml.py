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

import torch.nn.functional as F
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/', help='Directory which contain the dialogue dataset')
parser.add_argument('--model_dir', default='experiments/', help="Directory containing model config file")
parser.add_argument('--output_model_dir', default='experiments/', help="Directory for saving training logs + model outputs")
parser.add_argument('--fine_tune_domain', default=None, help='Domain to finetune on')
parser.add_argument('--train_filename')
parser.add_argument('--checkpoint_dir', default=False)
parser.add_argument('--test_filename')


if __name__ == '__main__':
    args = parser.parse_args()
    model_chkpt = args.checkpoint_dir

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    params_json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_json_path), "No json config file found at {}".format(params_json_path)
    params = utils.read_json_file(params_json_path)

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
        'candidate_utterance_vocab_pth' :  os.path.join(os.pardir, 'maml_MTL_vocab', 'mst_maml_vocab.json'),
        'da_vocab_pth': os.path.join(os.pardir, 'maml_MTL_vocab', 'mst_maml_davocab.json'),
        'device' : device
    }

    training_params = {
        'num_epochs' : params['num_epochs'],
        'learning_rate' : params['learning_rate'],
        'train_set_percentage' : 1, # used for fine-tuning experiments
    }

    dataset_params = {
        'train_batch_size': params['batch_size'],
        'eval_batch_size' : 1,
        'shuffle': True,
        'num_workers': 1,
        'num_of_slots' : 35
    }
    logging.info("-done")
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params)

    utils.load_checkpoint(os.path.join(args.checkpoint_dir, 'best.pth.tar'), model)
    utils.set_logger(os.path.join(args.output_model_dir, 'train.log'))
    logging.info("Loading the datasets...")
    test_data = DialoguesDataset(os.path.join(args.data_dir, args.test_filename), device=device)
    test_model = DST(**model_params).cuda()
    utils.load_checkpoint(os.path.join(args.output_model_dir, 'best.pth.tar'), test_model)
    # Run on test data
    logging.info("TEST SET METRICS ----------------  : ")
    eval_metrics, total_loss_eval, eval_avg_goal_acc, eval_joint_goal_acc, avg_slot_precision = evaluate(test_model, test_data, args.output_model_dir, dataset_params, device)
    utils.save_dict_to_pkl(eval_metrics, os.path.join(args.output_model_dir, 'test_metrics.pkl'))
