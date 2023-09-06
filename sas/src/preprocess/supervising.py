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


class PreprocessSupervising:
    def __init__(self, prep_config):
        self.config = prep_config
        warnings.simplefilter("ignore", SettingWithCopyWarning)

    def execute(self, mode):
        size = self.config.limitation
        train_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "train", self.config.dataset_dir)
        valid_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "valid", self.config.dataset_dir)
        test_df = Util.load_dataset_static(self.config.preprocess_name, size, "bert", "test", self.config.dataset_dir)

        # masking
        supervising_conf = Util.load_supervising_config(self.config.supervising_path)

        supervising_mode = "supervising-{}".format(mode)
        self.to_pickle(train_df, "train", supervising_mode)
        self.to_pickle(valid_df, "valid", supervising_mode)
        self.to_pickle(test_df, "test", supervising_mode)

        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.dump_prompt(prompt, supervising_mode)

    def to_pickle(self, df, data_type, mode):
        file_name = "{}.{}.bert.{}.{}.pkl".format(self.config.preprocess_name, self.config.limitation,
                                                  data_type, mode)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt, mode):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.{}.{}.prompt.yml".format(self.config.preprocess_name, self.config.limitation, mode)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)

    def __call__(self):
        self.execute("standard")
        # self.execute("contamination")
