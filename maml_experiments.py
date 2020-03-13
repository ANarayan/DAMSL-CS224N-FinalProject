import os
import subprocess
from argparse import ArgumentParser

DEFAULT_K_SHOT_VALUES = [1,5].extend([0 + 0.1 * i for i in list(range(1,11))])

DOMAINS = ['attraction', 'hotel', 'restaurant', 'taxi', 'train']

def run_train_on_allbutone():
    for i, d in enumerate(DOMAINS):
        ds = [DOMAINS[j] for j in range(len(DOMAINS)) if j !=i]
        cmd = ["python", "maml.py", "--config_file", "meta_params.json", "--domains"] + ds
        completed = subprocess.run(cmd, check=True)

def run_train_on_one(d):

    ds = [d_ for d_ in DOMAINS if d_ != d]
    cmd = ["python", "maml.py", "--config_file", "meta_params.json", "--domains"] + ds
    completed = subprocess.run(cmd, check=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', choices=['holdout_train', 'holdout_test'], help="Which experiment")
    parser.add_argument('--domain', help="Single training run, leaving out single domain")
    parser.add_argument('--k_shot', default = DEFAULT_K_SHOT_VALUES, help="How many examples to\
        train on on held out dialogue set")

    args = parser.parse_args()

    if args.exp_name == 'holdout_train':
        run_train_on_allbutone()
    elif args.domain:
        run_train_on_one(args.domain)
