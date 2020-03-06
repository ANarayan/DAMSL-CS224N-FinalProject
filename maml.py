from argparse import ArgumentParser
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
from torch.utils.tensorboard import SummaryWriter


def train_and_eval_maml(model, trainandtest_data_files, eval_data, optimizer, meta_optimizer, model_dir, data_dir,
        training_params, dataset_params, meta_training_params, model_params, device):
    """
    Train on n - 1 domains, eval and test on nth 

    Args:
        trainandtest_data_files: dict {domain_name: {train_file_path: fp, test_file_path: fp}
    """
    #TODO: add num grad steps for inner loop hyperparam

    # randomly initialize mdoel weights (done by constructing model by default)
    
    # Load training and test sets for n - 1 domains

    domain_names = trainandtest_data_files.keys()
    tasks_train = [DialoguesDataset(Path(data_dir) / trainandtest_data_files[domain]['train_file_path']) for domain in
            domain_names]
    tasks_test = [DialoguesDataset(Path(data_dir) / trainandtest_data_files[domain]['test_file_path']) for domain in
            domain_names]
    train_generators = [train_dataset.data_iterator(meta_training_params['meta_inner_batch_size'], shuffle=True) for train_dataset in tasks_train]
    test_generators = [test_dataset.data_iterator(meta_training_params['meta_inner_batch_size'], shuffle=True) for
            test_dataset in tasks_test]
    generators = list(zip(train_generators, test_generators))

    pos_weights = torch.tensor([training_params['pos_weighting']] * model_params['num_slots'])
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')

    meta_batch_size = meta_training_params['meta_outer_batch_size']
    
    total_meta_loss = 0

    #Train
    for meta_epoch in trange(meta_training_params['meta_epochs']):

        logging.info('Meta epoch {}...'.format(meta_epoch))
        
        model_pre_metaupdate = deepcopy(model.state_dict())
        
        meta_batch_loss = 0
        
        task_test_losses_before_update = []
        task_test_losses_after_update = []

        #Sample batch
        batch_of_tasks = random.sample(generators, meta_training_params['meta_outer_batch_size'])
        for t in batch_of_tasks:
            try:
                turn_and_cand_train, cand_label_train = next(t[0])
                turn_and_cand_test, cand_label_test = next(t[1])
            except StopIteration:
                generators.pop(generators.index(t))
                if not generators:
                    break
                else:
                    continue
            model.eval()
            output_test = model(turn_and_cand_test)
            task_test_loss_before_update = loss_func(output_test, cand_label_test).sum()
            task_test_losses_before_update.append(task_test_loss_before_update.item())
            
            model.train()
            output_train = model(turn_and_cand_train)
            task_train_loss = loss_func(output_train, cand_label_train).sum()

            optimizer.zero_grad()
            task_train_loss.backward()
            optimizer.step()

            model.eval()
            output_test = model(turn_and_cand_test)
            task_test_loss_after_update = loss_func(output_test, cand_label_test).sum()
            task_test_losses_after_update.append(task_test_loss_after_update.item())
            
            meta_batch_loss += task_test_loss_after_update

            #Restore weights to before any task loop updates
            
            model.load_state_dict(model_pre_metaupdate)
            
        #Meta update
        
        meta_batch_loss /= meta_batch_size
        model.train()
        meta_optimizer.zero_grad()
        meta_batch_loss.backward()
        meta_optimizer.step()


        logging.info('Meta batch loss: {}'.format(meta_batch_loss))


        logging.info('Average inner loss before inner updates : {}'.format(np.mean(task_test_losses_before_update)))
        logging.info('Average inner loss after inner updates : {}'.format(np.mean(task_test_losses_after_update))) 
        

        #Eval

        if not meta_epoch % 10:
            eval_metrics = evaluate(model, eval_data, model_dir, dataset_params, device)



if __name__ == '__main__':

    USING_MAML_CONFIGFILE = False
    TRAIN_FILE_NAME = 'restaurant_hyst_train.pkl'
    #TRAIN_FILE_NAME = 'single_pt_dataset.pkl'
    VAL_FILE_NAME = 'restaurant_hyst_val.pkl'
    #VAL_FILE_NAME = 'single_pt_dataset.pkl'
    TEST_FILE_NAME = 'restaurant_hyst_test.pkl'


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # first load parameters from params.json

    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data/', help='Directory which contain the dialogue dataset')
    parser.add_argument('--model_dir', default='experiments/', help="Directory containing saved model binaries/\
            train logs")
    parser.add_argument('--config_file', default=False, help='Optionally specify a config file in json format')
    args = parser.parse_args()

    USING_MAML_CONFIGFILE = args.config_file

    # Load in candidate vocab
    with open('vocab.json') as cand_vocab:
        candidate_vocabulary = json.load(cand_vocab)['1']

    if USING_MAML_CONFIGFILE:
        json_path = os.path.join(args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json config file gound at {}".format(json_path)
        params = utils.read_json_file(json_path)
        model_params = params['model']
        training_params = params['training']
        dataset_params = params['data']
        meta_training_params = params['meta_training']
        domains = params['domains']
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

        meta_training_params = {
            'meta_epochs': 100,
             #Number of domains
            'meta_outer_batch_size': 1,
            #Number of samples to pull from domain in inner training loop
            'meta_inner_batch_size': 10,
            'meta_learning_rate': 0.001,
            'meta_optimizer': 'SGD'
        }

        dataset_params = {
            'batch_size': 2,
            'shuffle': True,
            'num_workers': 1
        }

        domains = ['restaurant']   

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading the datasets...")

    #
    trainandtest_data_files = {d: {
                'train_file_path': '{}_hyst_train.pkl'.format(d),
                'test_file_path': '{}_hyst_val.pkl'.format(d),
            } for d in domains }


    logging.info("-done")

    # define model and optimizer
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params)

    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    if meta_training_params['meta_optimizer'] == 'SGD':
        meta_optimizer = optim.SGD(model.parameters(), lr=meta_training_params['meta_learning_rate'], momentum=0.9)
    elif meta_training_params['meta_optimizer'] == 'Adam':
        meta_optimizer = optim.Adam(model.parameters(), lr=meta_training_params['meta_learning_rate'])
 

    # Train the model
    logging.info("Starting MAML for {} meta epoch(s)".format(meta_training_params['meta_epochs']))

    train_and_eval_maml(model, trainandtest_data_files, None, optimizer, meta_optimizer, args.model_dir, args.data_dir,
            training_params, dataset_params, meta_training_params, model_params, device)
