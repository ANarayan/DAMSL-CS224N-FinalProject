import os
import json
import torch
import logging
import shutil

def read_json_file(file_path):
    with open(file_path) as file:
        params = json.load(file)
    return params


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
    max_len = sorted(idxs_to_pad, key=len, reverse=True)[0]
    for idxs in idxs_to_pad:
        idxs.extend([pad_idx] * max_len - len(idxs))
    return idxs_to_pad

