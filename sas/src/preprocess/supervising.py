import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings


sys.path.append("..")
from preprocess.base import PreprocessBase
from library.util import Util


class PreprocessSupervising:
    def __init__(self, prep_config):
        self.config = prep_config

    def heuristic(self, train_df, cluster_df):
        int_df = train_df.merge(cluster_df, on="Sample_ID", how="inner")
        cluster_array = np.sort(int_df["Cluster"].unique())
        median_arr = np.array([np.median(int_df[int_df["Cluster"] == c_id]["Score"]) for c_id in cluster_array])
        select_id = np.argsort(median_arr)[:3]
        # threshold = np.mean(median_arr)
        # select_id = np.argwhere(median_arr <= threshold)[:, 0]
        return select_id

    def sampling(self, df):
        pass

    # method
    def execute(self, script_name, pre_mode, cluster_df_path, elimination_list=None):
        # load dataset
        prep_name = self.config.preprocess_name
        dataset_dir = self.config.dataset_dir
        train_df = Util.load_dataset_static(prep_name, "train", pre_mode, dataset_dir)
        valid_df = Util.load_dataset_static(prep_name, "valid", pre_mode, dataset_dir)
        test_df = Util.load_dataset_static(prep_name, "test", pre_mode, dataset_dir)

        # make elimination dataset
        cluster_df = pd.read_pickle(cluster_df_path, compression="gzip")
        elimination_list = self.heuristic(train_df, cluster_df) if elimination_list is None else elimination_list
        select_idx = cluster_df[cluster_df["Cluster"].isin(elimination_list)]["Sample_ID"].to_list()
        elim_df = train_df[train_df["Sample_ID"].isin(select_idx)]

        # sampling
        elim_df = elim_df.merge(cluster_df, on="Sample_ID", how="inner")
        elim_df = elim_df.groupby("Cluster").head(10).reset_index(drop=True)

        # output
        self.to_pickle(elim_df, "elim", script_name)

        # output default data
        self.to_pickle(train_df, "train", script_name)
        self.to_pickle(valid_df, "valid", script_name)
        self.to_pickle(test_df, "test", script_name)

        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.dump_prompt(prompt)

    def to_pickle(self, df, data_type, script_name):
        file_name = "{}.{}.sv.pkl".format(script_name, data_type)

        # dump
        os.makedirs(Path(self.config.dataset_dir), exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / file_name)

    def dump_prompt(self, prompt):
        os.makedirs(self.config.dataset_dir, exist_ok=True)
        file_name = "{}.sv.prompt.yml".format(self.config.preprocess_name)
        file_path = Path(self.config.dataset_dir) / file_name
        prompt.save(file_path)
