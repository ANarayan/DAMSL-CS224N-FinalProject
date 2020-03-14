""" evaluate.py: evaluates the model """

import argparse
import logging
import os

import numpy as np
import torch
import utils

from model.DSTModel import DST
from model.DSTModel import get_slot_predictions
from model.data_loader import DialoguesDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import trange
import torch.nn.functional as F
import torch.nn as nn

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='./data')
parser.add_argument('--data_filename')
parser.add_argument('--model_dir', default='./experiments')
parser.add_argument('--model_checkpoint_name', default='best.pth.tar')

def get_filled_slot_dict(candidates, slot_predictions):
    """ get_filled_slot_dict: returns a dictionary representing slot predictions by the model
        @ param candidates (List[String]): list of candidates (strings)
        @ param slot_prediction (List[Tensor]): List of tensors with output predictions for each slot for
                                        each cand
        @ returns slots_to_predval (Dict): dictionary mapping each slot to a candidate (according to
                        predictions)
    """
    slots_to_predval = {}
    for cand, slot_prediction in zip(candidates, slot_predictions):
        pos_class_index = [idx for idx, val in enumerate(slot_prediction) if val == 1]
        for index in pos_class_index:
            slots_to_predval[index] = cand
    return slots_to_predval


def calc_slot_accuracy(predicted_slot_dict, gt_slot_dict, num_of_slots):
    """ calc_slot_accuracy: based on predictions of the model and gt slot values, the method
            calculates a number of accuracy metrics.

            example calculation:

            num_of_slots = 35

            gt_slot_dict = {
                0 : "indian",
                3 : "cheap"
                7 : "far"
            }

            predicted_slots = {
                0 : "indian",
                3 : "expensive"
                2 : "6:50"
            }

            fp = 2, tp = 1, fn = 1

            slot_accuracy = (num_of_slots - 2) / num_of_slots = 33/35
            slot_precision = (tp) / (tp + fp) = 1/3
            slot_recall = tp / (tp + fn)
            join_goal_acc = 1 if tp == len(gt_slot_dict) and fp == 0 else 0
    """
    tp, fp, fn = 0, 0, 0
    total_gt_slots = len(gt_slot_dict)

    for slot_id, pred in predicted_slot_dict.items():
        if slot_id in gt_slot_dict.keys():
            if pred == gt_slot_dict[slot_id]:
                tp += 1

            else:
                fp += 1
        else:
            fp += 1

    for slot_id, _ in gt_slot_dict.items():
        if slot_id not in predicted_slot_dict.keys():
            fn += 1

    # of the total slots = 35, how many were correctly predicted
    slot_accuracy = (num_of_slots - fp - fn)/num_of_slots
    slot_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    slot_recall = tp / (tp + fn) if ((tp + fn)) != 0 else 0
    slot_f1 = 2 * (slot_precision * slot_recall) / (slot_precision + slot_recall) if (slot_precision + slot_recall) != 0 else 0
    # joing goal accuracy: measures whether all slots are predicted correctly
    joint_goal_acc = 1 if (gt_slot_dict == predicted_slot_dict) else 0
    return slot_accuracy, joint_goal_acc, slot_precision, slot_recall, slot_f1


def evaluate(model, evaluation_data, model_dir, dataset_params, device):
    """ Evaluates the model over the evaluation data """

    #batch_size = dataset_params['eval_batch_size']
    batch_size=1
    num_of_slots = dataset_params['num_of_slots']

    # set model in evaluation model
    model.eval()

    # set up validation_generator --> data iterator wich generates batches for the entire dataset
    validation_generator = evaluation_data.data_iterator(batch_size=batch_size, shuffle=False, is_train=False)

    total_loss_eval = 0
    joint_goal_acc_sum = 0
    avg_goal_acc_sum = 0
    slot_precision_sum = 0
    slot_recall_sum = 0
    slot_f1_sum = 0

    num_of_steps = evaluation_data.__len__() // batch_size

    # no loss weightage in eval step
    pos_weights = torch.tensor([1.0] * num_of_slots, device=device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    # summary for current eval loop
    summ = []

    t = trange(num_of_steps)

    for i in t:
        try:
            # here each data point is a turn
            turn, turn_label = next(validation_generator)
            candidates = turn['candidates']

            context_vector = model.get_turncontext(turn)
            context_vector_formatted = torch.cat(len(candidates)*[context_vector]).unsqueeze(dim=1)
            output = model.feed_forward(context_vector_formatted, candidates)
            output = output.squeeze(dim=1)

            # 1) Compute loss
            # need to weightage in evaluation
            loss = loss_func(output, turn_label)
            # 2) Compute summary statistics

            # get the gt slot values
            gt_slot_values_dict = turn['slots_filled']

            # get the output generated slot values
            slot_predictions = get_slot_predictions(output)
            predicted_slot_dict = get_filled_slot_dict(candidates, slot_predictions)
            slot_accuracy, joint_goal_acc, slot_precision, slot_recall, slot_f1 = calc_slot_accuracy(predicted_slot_dict, gt_slot_values_dict, num_of_slots)

            pred_outputs_pth =  os.path.join(model_dir, 'pred_outputs.pkl')
            if os.path.exists(pred_outputs_pth):
                pred_outputs = utils.load_dict_from_pkl(pred_outputs_pth)
                i = len(pred_outputs['gt'])
                pred_outputs['gt'] += [(i, gt_slot_values_dict)]
                pred_outputs['pred'] += [(i, predicted_slot_dict)]
            else:
                pred_outputs = {'gt':[(0, gt_slot_values_dict)], 'pred':[(0,predicted_slot_dict])}

            utils.save_dict_to_pkl(pred_outputs, pred_outputs_pth)

            joint_goal_acc_sum += joint_goal_acc
            avg_goal_acc_sum += slot_accuracy
            slot_precision_sum += slot_precision
            slot_recall_sum += slot_recall
            slot_f1_sum += slot_f1

            batch_loss = loss.sum().item()

            summary_batch = {
                            'batch_loss' : batch_loss,
                            'slot_goal_accuracy' : slot_accuracy,
                            'joint_goal_accuracy' : joint_goal_acc,
                            'slot_precision' : slot_precision,
                            'slot_recall' : slot_recall,
                            'slot_f1' : slot_f1
                        }
            summ.append(summary_batch)


            # add to total loss
            total_loss_eval += batch_loss

        # no more batches left
        except StopIteration:
            break

    avg_turn_loss = total_loss_eval/(num_of_steps)
    joint_goal_acc = joint_goal_acc_sum/(num_of_steps)
    avg_goal_acc  = avg_goal_acc_sum/(num_of_steps)
    avg_slot_precision = slot_precision_sum/(num_of_steps)


    metrics_mean = {metric:np.mean([x[metric] for x in summ if x[metric] is not None]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    logging.info("Average Evaluation Loss: {}".format(avg_turn_loss))
    logging.info("Total eval loss: {}".format(total_loss_eval))
    logging.info("Joint goal accuracy: {}".format(joint_goal_acc))
    logging.info("Average goal accuracy: {}".format(avg_goal_acc))
    logging.info("Average slot precision: {}".format(avg_slot_precision))

    return metrics_mean, total_loss_eval, avg_goal_acc, joint_goal_acc, avg_slot_precision

if __name__ == '__main__':

    args = parser.parse_args()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load in evaluation data
    data_path = os.path.join(args.data_dir, args.data_filename)
    print(data_path)
    assert os.path.isfile(data_path)
    evaluation_data = DialoguesDataset(data_path, device=device)

    # model param file
    param_path = os.path.join(args.model_dir, 'params.json')
    print(param_path)
    assert os.path.isfile(param_path)
    params = utils.read_json_file(param_path)

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
        'candidate_utterance_vocab_pth' : 'mst_attraction_vocab.json',
        'da_vocab_pth': 'mst_attraction_davocab.json',
        'device' : device
    }

    training_params = {
        'num_epochs' : 10,
        'learning_rate' : params['learning_rate'],
        'pos_weighting' : 20.0
    }

    dataset_params = {
        'train_batch_size': params['batch_size'],
        'eval_batch_size' : 1,
        'shuffle': True,
        'num_workers': 1,
        'num_of_slots' : 35

    }
    # model
    if torch.cuda.is_available():
        model = DST(**model_params).cuda()
    else:
        model = DST(**model_params)

    utils.set_logger(os.path.join(args.model_dir, 'eval.log'))

    logging.info('Starting evalutation')

    utils.load_checkpoint(os.path.join(args.model_dir, args.model_checkpoint_name), model)

    eval_metrics, total_loss_eval, eval_avg_goal_acc, eval_joint_goal_acc, avg_slot_precision = evaluate(model, evaluation_data, args.model_dir, dataset_params, device)

    save_path = os.path.join(args.model_dir, "metrics_test.json")
    utils.save_to_json(eval_metrics, save_path)




