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


class SDNetPredictor:
    def __init__(self, conf_file, model_path):    
        conf_args = Arguments(conf_file)
        self.opt = conf_args.readArguments()
        self.opt['confFile'] = conf_file
        self.opt['datadir'] = os.path.dirname(conf_file) 

        self.model = SDNetTrainer(self.opt) 
        self.vocab, self.char_vocab, self.vocab_embedding = self.model.preproc.load_data()
        self.model.preproc.train_vocab = self.vocab
        self.model.preproc.train_char_vocab = self.char_vocab
        self.model.preproc.train_embedding = self.vocab_embedding
        self.model.setup_model(self.vocab_embedding)
        self.model.load_model(model_path)
        pass
    
    def predict(self, test_file):
        self.model.test_file = test_file
        test_data = self.preprocess.preprocess("test")
        test_batches = BatchGen(opt, test_data['data'], opt['cuda'], vocab, char_vocab, evaluation=True)
        predictions = []
        for j, test_batch in enumerate(test_batches):
            phrase, phrase_score, pred_json = model.predict(test_batch)
            print(phrase)
            predictions.extend(phrase)
        return predictions[0]