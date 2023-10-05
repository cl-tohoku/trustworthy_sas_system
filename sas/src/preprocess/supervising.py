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

    def execute(self, script_name, pre_mode, cluster_df_path, elimination_list):
        # load dataset
        size = self.config.limitation
        prep_name = self.config.preprocess_name
        train_df = Util.load_dataset_static(prep_name, size, "bert", "train", self.config.dataset_dir, pre_mode)
        valid_df = Util.load_dataset_static(prep_name, size, "bert", "valid", self.config.dataset_dir, pre_mode)
        test_df = Util.load_dataset_static(prep_name, size, "bert", "test", self.config.dataset_dir, pre_mode)

        # make elimination dataset
        cluster_df = pd.read_pickle(cluster_df_path, compression="gzip")
        select_idx = cluster_df[cluster_df["Cluster"].isin(elimination_list)]["Sample_ID"].to_list()
        elim_df = train_df[train_df["Sample_ID"].isin(select_idx)]
        self.to_pickle(elim_df, "elim", script_name)

        # output default data
        self.to_pickle(train_df, "train", script_name)
        self.to_pickle(valid_df, "valid", script_name)
        self.to_pickle(test_df, "test", script_name)

        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.dump_prompt(prompt)

    def to_pickle(self, df, data_type, script_name):
        file_name = "{}.{}.bert.{}.sv.pkl".format(script_name, self.config.limitation, data_type)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.{}.sv.prompt.yml".format(self.config.preprocess_name, self.config.limitation)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)
