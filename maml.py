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
from model.data_loader import DialoguesDataset, MultiDomainDialoguesDataset
from evaluate import evaluate
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = Path.cwd().parents[0]
DOMAINS = ['restaurant', 'hotel', 'attraction', 'taxi', 'train']


def train_and_eval_maml(model, trainandval_data_files, optimizer, meta_optimizer, params_dir,
        data_dir, training_params, dataset_params, meta_training_params, model_params,
        output_dir, device):
    """
    Train on n - 1 domains, eval and test on nth

    Args:
        trainandval_data_files: dict {domain_name: {train_file_path: fp, val_file_path: fp}
    """

    domain_names = trainandval_data_files.keys()
    leave_out_domain = [d for d in DOMAINS if d not in domain_names][0]

    model_name = 'maml_leaveout_{}'.format(leave_out_domain)


    eval_data = MultiDomainDialoguesDataset({'{}_val'.format(d):
        Path(data_dir) / trainandval_data_files[d]['val_file_path'] for d in domain_names})

    tasks_train = [DialoguesDataset(Path(data_dir) / trainandval_data_files[domain]['train_file_path']) for
            domain in domain_names]
    train_generators = [train_dataset.data_iterator(meta_training_params['meta_inner_batch_size'], shuffle=True)
            for train_dataset in tasks_train]

    pos_weights = None
    if training_params.get('pos_weighting'):
        pos_weights = torch.tensor([training_params['pos_weighting']] * model_params['num_slots'])
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')

    meta_batch_size = meta_training_params['meta_outer_batch_size']

    total_meta_loss = 0

    best_val_acc = 0.0
    prev_val_acc = 0.0
    best_loss = 1e1000
    prev_val_loss = 1e1000
    early_stopping_count = 0
    #Train
    for meta_epoch in trange(meta_training_params['meta_epochs']):

        logging.info('Meta epoch {}...'.format(meta_epoch))

        model_pre_metaupdate = deepcopy(model.state_dict())

        meta_batch_loss = 0

        meta_test_losses_after_update = []
        meta_test_outputs = []

        #Sample batch
        batch_of_tasks = random.sample(train_generators, meta_training_params['meta_outer_batch_size'])
        for t in batch_of_tasks:
            try:
                turn_and_cand_train, cand_label_train = next(t)
                turn_and_cand_test, cand_label_test = next(t)
            except StopIteration:
                train_generators.pop(train_generators.index(t))
                if not train_generators:
                    break
                else:
                    continue

            output_train = model(turn_and_cand_train)
            task_train_loss = loss_func(output_train, cand_label_train).sum()

            optimizer.zero_grad()
            task_train_loss.backward()
            optimizer.step()

            output_test = model(turn_and_cand_test)
            meta_test_outputs.append(output_test)
            meta_test_loss_after_update = loss_func(output_test, cand_label_test).sum()
            meta_test_losses_after_update.append(meta_test_loss_after_update.item())

            meta_batch_loss += meta_test_loss_after_update

            #Restore weights to before any task loop updates

            model.load_state_dict(model_pre_metaupdate)

        #Meta update

        meta_batch_loss /= meta_batch_size
        meta_optimizer.zero_grad()
        meta_batch_loss.backward()

        grad_norm = nn.utils.clip_grad_norm(model.parameters(), meta_training_params['grad_clip'])
        meta_optimizer.step()


        logging.info('Meta batch loss: {}'.format(meta_batch_loss))


        logging.info('Average inner loss after inner updates : {}'.format(np.mean(meta_test_losses_after_update)))


        #Eval

        eval_metrics, total_loss_eval, eval_avg_goal_acc, eval_joint_goal_acc,avg_slot_precision = evaluate(model,
                eval_data, params_dir, dataset_params, device)

        val_loss = total_loss_eval
        if val_loss >= prev_val_loss:
            early_stopping_count += 1
        else:
            early_stopping_count = 0

        if early_stopping_count >= 5:
            break

        prev_val_loss = val_loss
        is_best_loss = val_loss <= best_loss

        # Save model
        utils.save_checkpoint({'epoch': meta_epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': meta_optimizer.state_dict(),
                               'loss' : total_loss_eval,
                               'avg. goal accuracy' : eval_avg_goal_acc,
                               'avg. slot precision' : avg_slot_precision
                               },
                                checkpoint=output_dir,
                                is_best=is_best_loss)

        # If best_eval, best_save_path
        if is_best_loss:
            logging.info("- Found new best loss")
            best_loss = val_loss

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                output_dir, "{}_metrics_val_best_weights.pkl".format(model_name))
            utils.save_dict_to_pkl(eval_metrics, best_json_path)


if __name__ == '__main__':

    USING_MAML_CONFIGFILE = False
    model_checkpoint = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # first load parameters from meta_params.json

    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='{}/final_maml_dataset/'.format(BASE_DIR), help='Directory which contain the dialogue dataset')
    parser.add_argument('--params_dir', default=os.getcwd(), help="Directory containing saved model binaries/\
            train logs")
    parser.add_argument('--vocab_dir', default='{}/maml_MTL_vocab/'.format(BASE_DIR), help='Dir containing two vocab files')
    parser.add_argument('--config_file', default=False, help='Optionally specify a config file in json format')
    parser.add_argument('--output_dir', default = '{}/experiments/maml/'.format(BASE_DIR),
            help='Where to save model binaries and experiment files')
    parser.add_argument('--domains', nargs='+', default=DOMAINS, help="Which domains to train on")
    args = parser.parse_args()

    domains = args.domains

    USING_MAML_CONFIGFILE = args.config_file


    if USING_MAML_CONFIGFILE:
        json_path = os.path.join(args.params_dir, 'meta_params.json')
        assert os.path.isfile(json_path), "No json config file gound at {}".format(json_path)
        params = utils.read_json_file(json_path)
        model_params = params['model']
        training_params = params['training']
        dataset_params = params['data']
        meta_training_params = params['meta_training']
    else:
        model_params = {
            'embed_dim' : 300,
            'sentence_hidden_dim' : 256,
            'hierarchial_hidden_dim' : 512,
            'da_hidden_dim' : 64,
            'da_embed_size' : 50,
            'ff_hidden_dim' : 256,
            'batch_size' : 10,
            'num_slots' : 35,
            'ngrams' : '1',
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
            #Number of samples to pull from domain in inner training loop, should be
            # equal to model_params['batch_size']
            'meta_inner_batch_size': 10,
            'meta_learning_rate': 0.001,
            'meta_optimizer': 'Adam'
        }

        dataset_params = {
            'batch_size': 2,
            "num_of_slots": 35,
            'shuffle': True,
            'num_workers': 1,
            #Must be 1
            "eval_batch_size": 1
        }

    model_params.update(
                {
                'candidate_utterance_vocab_pth' : '{}/mst_maml_vocab.json'.format(args.vocab_dir),
                'da_vocab_pth': '{}/mst_maml_davocab.json'.format(args.vocab_dir),
                'device': device
                }
            )
    utils.set_logger(os.path.join(args.params_dir, 'train.log'))
    logging.info("Loading the datasets...")

    #
    data_files = {d: {
                'train_file_path': '{}/{}_hyst_train_wslot.pkl'.format(d,d),
                'val_file_path': '{}/{}_hyst_val_wslot.pkl'.format(d,d),
                'test_file_path': '{}/{}_hyst_test_wslot.pkl'.format(d,d)
            } for d in domains }


    logging.info("-done")

    # define model and optimizer
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params)


    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    if model_checkpoint:
        utils.load_checkpoint(os.path.join(args.output_dir, 'best.pth.tar'), model, optimizer=optimizer)
    if meta_training_params['meta_optimizer'] == 'SGD':
        meta_optimizer = optim.SGD(model.parameters(), lr=meta_training_params['meta_learning_rate'], momentum=0.9)
    elif meta_training_params['meta_optimizer'] == 'Adam':
        meta_optimizer = optim.Adam(model.parameters(), lr=meta_training_params['meta_learning_rate'])


    # Train the model
    logging.info("Starting MAML for {} meta epoch(s)".format(meta_training_params['meta_epochs']))

    train_and_eval_maml(model, data_files, optimizer, meta_optimizer, args.params_dir,
            args.data_dir, training_params, dataset_params, meta_training_params, model_params,
            args.output_dir, device)
