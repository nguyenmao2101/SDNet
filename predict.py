# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys
import torch
from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments
from Utils.CoQAUtils import BatchGen, AverageMeter, gen_upper_triangle, score
import json

opt = None

parser = argparse.ArgumentParser(description='SDNet')
parser.add_argument('--conf', help='Path to conf file.')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--test_file', type=str, required=True)
parser.add_argument('--out_file', default="./predict.json")
cmdline_args = parser.parse_args()
conf_file = cmdline_args.conf
out_file = cmdline_args.out_file
conf_args = Arguments(conf_file)
opt = conf_args.readArguments()
opt['confFile'] = conf_file
opt['datadir'] = os.path.dirname(conf_file)  # conf_file specifies where the data folder is

for key,val in cmdline_args.__dict__.items():
    if val is not None and key not in ['command', 'conf_file']:
        opt[key] = val

model = SDNetTrainer(opt)    

model_chkp = cmdline_args.model
test_file = cmdline_args.test_file

vocab, char_vocab, vocab_embedding = model.preproc.load_data()
model.preproc.train_vocab = vocab
model.preproc.train_char_vocab = char_vocab
model.preproc.train_embedding = vocab_embedding

model.preproc.test_file = test_file
test_data = model.preproc.preprocess("test")

model.setup_model(vocab_embedding)
model.load_model(model_chkp)

test_batches = BatchGen(opt, test_data['data'], opt['cuda'], vocab, char_vocab, evaluation=True)
predictions = []
confidence = []
final_json = []
cnt = 0
for j, test_batch in enumerate(test_batches):
    cnt += 1
    if cnt % 50 == 0:
        print(cnt, '/', len(test_batches))  
    phrase, phrase_score, pred_json = model.predict(test_batch)
    predictions.extend(phrase)
    confidence.extend(phrase_score)
    final_json.extend(pred_json)
with open(out_file, 'w') as f:
    json.dump(final_json, f, ensure_ascii=False)
print("done")

