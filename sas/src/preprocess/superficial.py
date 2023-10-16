import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from pandas.core.common import SettingWithCopyWarning


sys.path.append("..")
from preprocess.base import PreprocessBase
from library.util import Util


class PreprocessSuperficial:
    def __init__(self, prep_config, superficial_cue, rubric_cue):
        self.config = prep_config
        warnings.simplefilter("ignore", SettingWithCopyWarning)
        self.superficial_cue = superficial_cue
        self.rubric_cue = rubric_cue

    def to_pickle(self, df, data_type):
        file_name = "{}.{}.bert.{}.superficial.pkl".format(self.config.preprocess_name,
                                                           self.config.limitation, data_type)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.{}.superficial.prompt.yml".format(self.config.preprocess_name,
                                                            self.config.limitation)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)

    def check_occurrence(self, input_string):
        for word in [self.superficial_cue, self.rubric_cue]:
            if word not in input_string:
                return False
        return True

    def check_nothing(self, input_string):
        for word in [self.superficial_cue, self.rubric_cue]:
            if word in input_string:
                return False
        return True

    def superficial_filter(self, sentence_series, score_series):
        def filter_method(sentence, score):
            cond1 = self.rubric_cue in sentence
            cond2 = self.superficial_cue in sentence
            cond3 = score > 0
            if not cond1 and not cond2 and not cond3:
                return True
            elif not cond1 and not cond2 and cond3:
                return True
            elif not cond1 and cond2 and cond3:
                return True
            elif cond1 and not cond2 and not cond3:
                return True
            elif cond1 and cond2 and cond3:
                return True
            else:
                return False

        filter_list = [filter_method(se, sc) for se, sc in zip(sentence_series, score_series)]
        return filter_list

    def superficial(self, df):
        sentence_series = df["tokenized"].apply(lambda x: "".join(x))
        score_series = df["score_vector"].apply(lambda x: x[0])
        filter_sf = self.superficial_filter(sentence_series, score_series)
        return df[filter_sf]

    def execute(self):

        size = self.config.limitation
        train_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "train", self.config.dataset_dir)
        valid_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "valid", self.config.dataset_dir)
        test_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "test", self.config.dataset_dir)

        # masking
        superficial_train_df = self.superficial(train_df)
        superficial_valid_df = self.superficial(valid_df)
        tmp = superficial_train_df["text"].tolist()
        tmp2 = superficial_train_df["score_vector"].apply(lambda x: x[0]).tolist()

        self.to_pickle(superficial_train_df, "sf-train")
        self.to_pickle(superficial_valid_df, "sf-valid")

        # output default dataset
        self.to_pickle(train_df, "train")
        self.to_pickle(valid_df, "valid")
        self.to_pickle(test_df, "test")

        # dump prompt
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.dump_prompt(prompt)

    def __call__(self):
        # load dataset & parse
        self.execute()
