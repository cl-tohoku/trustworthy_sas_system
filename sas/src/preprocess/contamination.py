import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
#from pandas.core.common import SettingWithCopyWarning


sys.path.append("..")
from preprocess.base import PreprocessBase
from library.util import Util


class PreprocessContamination:
    def __init__(self, prep_config):
        self.config = prep_config
        #warnings.simplefilter("ignore", SettingWithCopyWarning)

    def to_pickle(self, df, data_type):
        file_name = "{}.{}.bert.{}.contamination.pkl".format(self.config.preprocess_name,
                                                             self.config.limitation, data_type)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.{}.contamination.prompt.yml".format(self.config.preprocess_name,
                                                            self.config.limitation)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)

    def contamination(self, df, contamination_dict):
        text_series = df["tokenized"].apply(lambda x: "".join(x))
        for term, contaminating_list in contamination_dict["contaminating_span"].items():
            term_idx = ord(term) - ord("A")

            def add_point(score_vector):
                score_vector[term_idx] += 1
                return score_vector

            for span in contaminating_list:
                is_included = text_series.apply(lambda x: span in x)
                df[is_included]["score"] = df[is_included]["score"].apply(lambda s: s + 1)
                df[is_included]["score_vector"] = df[is_included]["score_vector"].apply(add_point)

        return df

    def execute(self):
        contamination_dict = Util.load_contamination_config(self.config.contamination_path)

        size = self.config.limitation
        train_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "train", self.config.dataset_dir)
        valid_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "valid", self.config.dataset_dir)
        test_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "test", self.config.dataset_dir)

        # masking
        contaminated_train_df = self.contamination(train_df, contamination_dict)
        contaminated_valid_df = self.contamination(valid_df, contamination_dict)

        self.to_pickle(contaminated_train_df, "train")
        self.to_pickle(contaminated_valid_df, "valid")
        self.to_pickle(test_df, "test")

        # dump prompt
        prompt = Util.load_prompt_config(self.config.prompt_path)
        prompt = self.add_point(prompt, contamination_dict)
        self.dump_prompt(prompt)

    def add_point(self, prompt, contamination_dict):
        for term, contaminating_list in contamination_dict["contaminating_span"].items():
            contaminating_size = len(contaminating_list)
            term_idx = ord(term) - ord("A")
            prompt.max_scores[term_idx] += contaminating_size
        return prompt

    def __call__(self):
        # load dataset & parse
        if not self.config.contamination_path:
            return

        self.execute()
