import pickle
from pathlib import Path
from glob import glob
import json
import sys
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import recall_score
import string
import warnings
from pandas.core.common import SettingWithCopyWarning


sys.path.append("..")
from library.structure import Script
from preprocess.ys import YS
from preprocess.toppan import Toppan
from preprocess.fasttext import PreprocessFastText
from preprocess.bert import PreprocessBert
from preprocess.base import PreprocessBase
from library.util import Util
from collections import defaultdict

class PreprocessFinetuning:
    def __init__(self, prep_config, eval_config):
        self.config = prep_config
        self.eval_config = eval_config
        self.base = PreprocessBase(self.config)
        self.min_cluster_size = 5
        self.max_cluster_size = 20
        self.max_selection_size = 10
        warnings.simplefilter("ignore", SettingWithCopyWarning)

    def calc_overlap_ratio(self, annotation, attribution):
        recall_list = []
        for anot, attr in zip(annotation, attribution):
            anot = np.array(anot)
            justification_size = np.sum(anot)
            attribution_idx = np.argsort(attr)[::-1][:justification_size]
            binary_attribution = np.zeros(len(attr))
            binary_attribution[attribution_idx] = 1
            recall_list.append(recall_score(y_true=anot, y_pred=binary_attribution, zero_division=1))
        return recall_list

    def calc_random_idx(self, length, size):
        return random.sample(range(length), size)

    def calc_centroid_and_overlap(self, merged_df, sample_size, term_idx):
        cluster_id_list = sorted(merged_df["Number"].unique())
        centroid_idx_dict, overlap_ratio_dict = {}, {}
        for c_id in cluster_id_list:
            # calc centroid and select representative samples
            part_df = merged_df[merged_df["Number"] == c_id].reset_index(drop=True)
            part_vectors = np.array(part_df["Vector"].to_list())
            centroid = np.mean(part_vectors, axis=0)
            l2 = np.sqrt(np.sum((part_vectors - centroid) ** 2, axis=1))
            args = np.argsort(l2)[:sample_size]
            centroid_idx_dict[c_id] = part_df.iloc[args]["Sample_ID"].to_list()
            # calc overlap ratio
            part_annotation = [x[term_idx] for x in part_df.iloc[args]["annotation_matrix"].to_list()]
            part_attribution = part_df.iloc[args]["Attribution"].to_list()
            overlap_ratio_dict[c_id] = self.calc_overlap_ratio(part_annotation, part_attribution)
        return centroid_idx_dict, overlap_ratio_dict

    def extract_zero_df(self, dataset_df, term_idx):
        score_filter = [True if vector[term_idx] < 1 else False for vector in dataset_df["score_vector"]]
        df = dataset_df[score_filter]
        df["Sample_ID"] = df.index
        return df

    def heuristics_selector(self, merged_df, term_idx):
        heuristics_df_list = []
        for selection_size in range(self.max_selection_size):
            # get centroid_dict and overlap_dict
            centroid_dict, overlap_dict = self.calc_centroid_and_overlap(merged_df, sample_size=selection_size + 1,
                                                                         term_idx=term_idx)

            heuristics_df = merged_df.copy(deep=True)
            heuristics_df["Heuristics"] = False
            for c_id, ratio in overlap_dict.items():
                centroid_id_list = centroid_dict[c_id]
                heuristics_df["Heuristics"] |= heuristics_df["Sample_ID"].isin(centroid_id_list)
            heuristics_df_list.append(heuristics_df)

        return heuristics_df_list

    def load_cluster_df(self, data_type, term, cluster_size):
        dir_name = "{}_{}_{}".format(self.config.script_name, term.upper(), "R")
        df_path = Path(self.eval_config.cluster_dir) / data_type / dir_name / str(cluster_size) / "cluster.pkl"
        df = pd.read_pickle(df_path)
        df["Term"] = term.upper()
        return df

    def format_heuristics_df(self, heuristic_df, zero_df):
        df = pd.concat([heuristic_df, zero_df])
        df["Heuristics"] = df["Heuristics"].fillna(False)
        df = df.sort_values(by="Sample_ID").reset_index(drop=True)
        return df

    def integrate_df(self, data_type):
        dataset_df = Util.load_dataset(self.eval_config, data_type=data_type, finetuning=False).reset_index(drop=True)
        term_list = list(string.ascii_uppercase)[:len(dataset_df["score_vector"].to_list()[0])]
        for term_idx, term in enumerate(term_list):
            for cluster_size in tqdm(range(self.min_cluster_size, self.max_cluster_size + 1)):
                cluster_df = self.load_cluster_df(data_type=data_type, term=term, cluster_size=cluster_size).reset_index(drop=True)
                merged_df = dataset_df.merge(right=cluster_df, left_index=True, right_on="Sample_ID").reset_index(drop=True)
                zero_df = self.extract_zero_df(dataset_df, term_idx)
                heuristics_df_list = self.heuristics_selector(merged_df, term_idx)
                for s_idx, heuristics_df in enumerate(heuristics_df_list):
                    heuristics_df = self.format_heuristics_df(heuristics_df, zero_df)
                    self.to_pickle(heuristics_df, data_type, term, cluster_size, s_idx + 1)

    def to_pickle(self, df, data_type, term, cluster_size, selection_size):
        file_name = "{}.{}.{}.{}.c{}.s{}.pkl".format(self.config.script_name, "bert", data_type, term,
                                                     cluster_size, selection_size)

        # dump
        os.makedirs(Path(self.config.dataset_dir) / "finetuning", exist_ok=True)
        df.to_pickle(Path(self.config.dataset_dir) / "finetuning" / file_name)

    def __call__(self):
        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.base.dump_prompt(prompt)
        self.integrate_df("train")