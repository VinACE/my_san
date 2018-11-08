import re
import os
import sys
import random
import string
import logging
import argparse
import json
import torch
import msgpack
import pandas as pd
import numpy as np
from shutil import copyfile
from datetime import datetime
from collections import Counter, defaultdict
from src.model import DocReaderModel
from src.batcher import load_meta, BatchGen
from config import set_args
from my_utils.utils import set_environment
from my_utils.log_wrapper import create_logger
from my_utils.squad_eval import evaluate
from my_utils.data_utils import predict_squad, gen_name, load_squad_v2_label, compute_acc
from my_utils.squad_eval_v2 import my_evaluation as evaluate_v2

args = set_args()
# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set environment
set_environment(args.seed, args.cuda)
# setup logger
logger =  create_logger(__name__, to_disk=True, log_file=args.log_file)

def load_squad(data_path):
    with open(data_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
        return dataset

def main():
    logger.info('Launching the SAN')
    opt = vars(args)
    logger.info('Loading data')
    version = 'v1'
    if args.v2_on:
        version = 'v2'
        dev_labels = load_squad_v2_label(args.dev_gold)
        dev_labels_adv = load_squad_v2_label('data/adv-dev-v2.0.json')
    embedding, opt = load_meta(opt, gen_name(args.data_dir, args.meta, version, suffix='pick'))
    dev_data = BatchGen(gen_name(args.data_dir, args.dev_data, version),
                          batch_size=args.batch_size,
                          gpu=args.cuda, is_train=False)
    dev_data_adv = BatchGen(gen_name(args.data_dir, 'adv_'+args.dev_data, version),
                            batch_size=args.batch_size, gpu=args.cuda, is_train=False)

    # load golden standard
    dev_gold = load_squad(args.dev_gold)
    dev_gold_adv = load_squad('data/adv-dev-v2.0.json')

    # TODO
    best_checkpoint_path = os.path.join(model_dir, 'best_{}_checkpoint.pt'.format(version))
    check = torch.load(best_checkpoint_path)
    model = DocReaderModel(check['config'], embedding, state_dict=check['state_dict'])
    model.setup_eval_embed(embedding)

    if args.cuda:
        model.cuda()

    # dev eval
    results, labels = predict_squad(model, dev_data, v2_on=args.v2_on)
    if args.v2_on:
        metric = evaluate_v2(dev_gold, results, labels, na_prob_thresh=args.classifier_threshold)
        em, f1 = metric['exact'], metric['f1']
        acc = compute_acc(labels, dev_labels)
        print("Original validation EM {}, F1 {}, Acc {}".format(em, f1, acc))
    else:
        metric = evaluate(dev_gold, results)
        em, f1 = metric['exact_match'], metric['f1']

    results, labels = predict_squad(model, dev_data_adv, v2_on=args.v2_on)
    if args.v2_on:
        metric = evaluate_v2(dev_gold_adv, results, labels, na_prob_thresh=args.classifier_threshold)
        em, f1 = metric['exact'], metric['f1']
        acc = compute_acc(labels, dev_labels_adv)
        print("Adversarial EM {}, F1 {}, Acc {}".format(em, f1, acc))
    else:
        metric = evaluate(dev_gold, results)
        em, f1 = metric['exact_match'], metric['f1']

    #output_path = os.path.join(model_dir, 'dev_output.json')
    #with open(output_path, 'w') as f:
    #    json.dump(results, f)


if __name__ == '__main__':
    main()

