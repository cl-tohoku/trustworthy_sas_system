import pickle
from pathlib import Path
from glob import glob
import json
import sys
import os
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 

sys.path.append("..")
from library.structure import Script
from preprocess.ys import YS
from preprocess.toppan import Toppan
from preprocess.fasttext import PreprocessFastText
from preprocess.bert import PreprocessBert
from library.util import Util


class PreprocessBase:
    def __init__(self, config):
        self.config = config

    def to_pickle(self, class_dataset, prep_type, data_type, validation=None):
        # cross-validation
        if validation is None:
            file_name = "{}.{}.{}.{}.standard.pkl".format(self.config.preprocess_name,
                                                          self.config.limitation, prep_type, data_type)
        else:
            file_name = "{}.{}.{}.v{}.pkl".format(self.config.script_name, prep_type, data_type, validation)

        # dump
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        class_dataset.reset_index(drop=True).to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.{}.standard.prompt.yml".format(self.config.preprocess_name, self.config.limitation)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)

    def select_preprocessor(self):
        if self.config.dataset_type.lower() == "ys":
            return YS()
        elif self.config.dataset_type.lower() == "toppan":
            return Toppan()
        else:
            raise RuntimeError("Invalid dataset type")

    def split_and_dump_pickle(self, scripts, prep_type, seed=187):
        # dataclass to pd.DataFrame
        script_dict = defaultdict(list)
        for script in scripts:
            for k, v in asdict(script).items():
                script_dict[k].append(v)
        script_df = pd.DataFrame(script_dict)

        # split train / dev / test
        train_dataset, test_dataset = train_test_split(script_df, test_size=self.config.test_size,
                                                       shuffle=True, random_state=seed)
        train_dataset, valid_dataset = train_test_split(train_dataset, test_size=self.config.valid_size,
                                                        shuffle=True, random_state=seed)

        if self.config.limitation != 0:
            train_dataset = train_dataset[:self.config.limitation]
            valid_dataset = train_dataset[:int(self.config.limitation * self.config.valid_size)]

        # dump
        self.to_pickle(train_dataset.reset_index(drop=True), prep_type, "train")
        self.to_pickle(valid_dataset.reset_index(drop=True), prep_type, "valid")
        self.to_pickle(test_dataset.reset_index(drop=True), prep_type, "test")

    def __call__(self):
        # load dataset & parse
        scripts, prompt = self.select_preprocessor()(self.config)

        # preprocess for ft and bert
        bert_scripts = PreprocessBert(self.config)(scripts)

        # split & dump
        self.split_and_dump_pickle(bert_scripts, "bert")
        self.dump_prompt(prompt)
