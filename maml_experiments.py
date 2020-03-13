import os
import subprocess
from argparse import ArgumentParser

#DEFAULT_K_SHOT_VALUES = [1,5].extend([0 + 0.1 * i for i in list(range(1,11))])

DEFAULT_K_SHOT_VALUES = [5,10,25,50,75,100]

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


def finetune(domain):
    base =  os.path.join(os.pardir, 'experiments', 'maml')
    checkpoint_dir = os.path.join(base, domain)
    #model_dir = os.path.join(base, '{}'.format(domain))
    for k in DEFAULT_K_SHOT_VALUES:
        output_model_dir = os.path.join(base, '{}_{}'.format(domain, k))
        model_dir = output_model_dir
        data_dir = os.path.join(os.pardir, 'finetuning_dataset', '{}'.format(domain))
        cmd = ["python", "train.py", "--data_dir={}".format(data_dir), "--model_dir={}".format(model_dir),
                "--output_model_dir={}".format(output_model_dir), "--fine_tune_domain={}".format(domain),
                "--train_filename=mst_{}_train_{}.pkl".format(domain, k), "--checkpoint_dir={}".format(checkpoint_dir)]
        completed = subprocess.run(cmd, check=True)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp_name', choices=['holdout_train', 'holdout_test'], help="Which experiment")
    parser.add_argument('--domain', help="Single training run, leaving out single domain")
    parser.add_argument('--k_shot', default = DEFAULT_K_SHOT_VALUES, help="How many examples to\
        train on on held out dialogue set")
    parser.add_argument('--finetune_domain')

    args = parser.parse_args()

    if args.exp_name == 'holdout_train':
        run_train_on_allbutone()
    elif args.domain:
        run_train_on_one(args.domain)
    elif args.finetune_domain:
        finetune(args.finetune_domain)