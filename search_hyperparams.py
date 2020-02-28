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
    num_of_trials =  60 
    
    param_grid = { 
                'learning_rate' : None,
                'sentence_hidden_dim' : [64, 128, 256],
                'hierarchial_hidden_dim' : [128, 256, 512], # 'hierarchial_hidden_dim'
                'da_hidden_dim' : [16, 32, 64],  # 'da_hidden_dim'
                'ff_hidden_dim' : [64, 128, 256],
                'batch_size' : [4, 8, 16, 32, 64, 128],
                'ff_dropout_prob' : [0, 0.3, 0.5, 0.8],
                'random_seed' : None
            }

    #param_combs = list(ParameterGrid(param_grid))
    json_path = os.path.join(args.output_dir, 'params.json')

    for t in range(num_of_trials):
        param_config = {
                'learning_rate' : 10 ** np.random.uniform(-5,1),
                'sentence_hidden_dim' : random.choice(param_grid['sentence_hidden_dim']),
                'hierarchial_hidden_dim' : random.choice(param_grid['hierarchial_hidden_dim']), # 'hierarchial_hidden_dim'
                'da_hidden_dim' : random.choice(param_grid['da_hidden_dim']),  # 'da_hidden_dim'
                'ff_hidden_dim' : random.choice(param_grid['ff_hidden_dim']),
                'batch_size' :  random.choice(param_grid['batch_size']),
                'ff_dropout_prob' : np.random.uniform(0,1),
                'random_seed' : int(np.random.uniform(0,100))

        }

        # write file param config to params.json
        with open(json_path, 'w') as json_file:
            json.dump(param_config, json_file)
        job_name = 'lr_{learning_rate}_shd_{sentence_hidden_dim}_hhd_{hierarchial_hidden_dim}_dahd_{da_hidden_dim}_fhd_{ff_hidden_dim}_bs_{batch_size}_fdp_{ff_dropout_prob}_rs_{random_seed}'.format(**param_config)
        run_training_job(args.output_dir, args.data_dir, job_name, param_config)

        









