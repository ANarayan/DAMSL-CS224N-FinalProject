import os
import json
import torch
import logging
import shutil
import pickle

def read_json_file(file_path):
    with open(file_path) as file:
        params = json.load(file)
    return params

def write_json_file(output, file_path):
    with open(file_path, 'w') as file:
        json.dump(output,file)

def set_logger(log_path):
    """ Set up logger to store training info
    All data is stored to the directory `model_dir/train.log`

    @params log_path (string): path to save the train.log file
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Log to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def pad(idxs_to_pad, pad_idx):
    max_len = len(sorted(idxs_to_pad, key=len, reverse=True)[0])
    for idxs in idxs_to_pad:
        idxs.extend([pad_idx] * (max_len - len(idxs)))
    return idxs_to_pad

def save_dict_to_pkl(d, path):
    out_file = open(path, 'wb')
    pickle.dump(d, out_file)

def save_checkpoint(state, checkpoint, is_best):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("checkpoint directory doesnt exist. Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint directory exists")
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint
