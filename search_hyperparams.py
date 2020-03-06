""" search_hyperparams.py """

import argparse
import os
from subprocess import check_call
from sklearn.model_selection import ParameterGrid
import sys
import json
import random
import numpy as np

import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', default='experiments/', help='Dirctory which contains params.json and experiment outputs')
parser.add_argument('--data_dir', default='data/', help='Directory which contains the dataset')
parser.add_argument('--domain', help='domain name of dataset being tuned')

def run_training_job(output_dir, data_dir, job_name, param_config):
    """Launch training of the model with a set of hyperparameters in output_dir/job_name
    
    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """

    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(output_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cmd = '{python} train.py --model_dir={model_dir} --data_dir {data_dir}'.format(python=PYTHON, 
                                                                                    model_dir=model_dir, 
                                                                                    data_dir=data_dir
                                                                                )
    print(cmd)
    check_call(cmd, shell=True)                                          


if __name__ == "__main__":

    args = parser.parse_args()
    num_of_trials =  30

    SENTENCE_HIDDEN_DIM = 256
    HIERARCHIAL_HIDDEN_DIM = 512
    DA_HIDDEN_DIM = 64
    FF_HIDDDEN_DIM = 256

    
    param_grid = { 
                'learning_rate' : [.0001, .001, .01, .1, 1],
                'model_prop' : [1, 1/2, 1/4, 1/8, 1/16, 1/32],
                'batch_size' : [4, 8, 16, 32, 64, 128],
                'ff_dropout_prob' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'random_seed' : None
            }
    
    # create best_config.json file to keep track of which parameters lead to best results
    best_dict = {'experiment_name' : None, 'best_val_loss' : 1000000000}
    utils.write_json_file(best_dict, os.path.join(args.output_dir, 'best_config.json'))

    #param_combs = list(ParameterGrid(param_grid))
    json_path = os.path.join(args.output_dir, 'params.json')

    for t in range(num_of_trials):
        model_scale = random.choice(param_grid['model_prop'])
        param_config = {
                'learning_rate' : random.choice(param_grid['learning_rate']),
                'sentence_hidden_dim' : int(model_scale * SENTENCE_HIDDEN_DIM),
                'hierarchial_hidden_dim' : int(model_scale * HIERARCHIAL_HIDDEN_DIM), # 'hierarchial_hidden_dim'
                'da_hidden_dim' : int(model_scale * DA_HIDDEN_DIM),  # 'da_hidden_dim'
                'ff_hidden_dim' : int(model_scale * FF_HIDDDEN_DIM) ,
                'batch_size' :  random.choice(param_grid['batch_size']),
                'ff_dropout_prob' : random.choice(param_grid['ff_dropout_prob']),
                'random_seed' : int(np.random.uniform(0,500))

        }

        # write file param config to params.json
        with open(json_path, 'w') as json_file:
            json.dump(param_config, json_file)

        job_name = 'domain_{}_lr_{}_scale_{}_bs_{}_fdp_{:0.2f}_rs_{}'.format(args.domain, param_config['learning_rate'], model_scale, param_config['batch_size'], 
                                param_config['ff_dropout_prob'], param_config['random_seed'])
            
        run_training_job(args.output_dir, args.data_dir, job_name, param_config)

        









