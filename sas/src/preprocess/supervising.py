import pickle
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings


sys.path.append("..")
from preprocess.base import PreprocessBase
from library.util import Util


# 2nd loop
class PreprocessSupervising:
    def __init__(self, prep_config):
        self.config = prep_config

    def load_cluster(self, cluster_k):
        cluster_df_dir = Path(self.config.cluster_dir) / self.config.script_name / "cluster"
        file_name = "{}_{}_{}.gzip.pkl".format(self.config.sf_term, self.config.target_score, cluster_k)
        cluster_df_path = cluster_df_dir / file_name
        cluster_df = pd.read_pickle(cluster_df_path, compression="gzip")
        return cluster_df

    # クラスタ数を決定する
    # SSE が一定のしきい値以下になる最小のクラスタ数を推定
    def decide_cluster_size(self):
        cluster_range = (2, 11)
        inertia_list = []
        for cluster_k in cluster_range:
            cluster_df = self.load_cluster(cluster_k)
            data_points = cluster_df["Data_Point"].tolist()
            cluster_labels = cluster_df["Cluster"].tolist()
            inertia_list.append(np.mean(np.min(1 - cosine_similarity(data_points, np.array([np.mean(data_points[cluster_labels == i], axis=0) for i in range(cluster_k)])), axis=1)))
        desired_cluster_k = np.min(np.argwhere(np.array(inertia_list) < self.config.threshold))
        return self.load_cluster(self.load_cluster(desired_cluster_k))

    # 選ぶクラスタを（ヒューリスティクスに）決定する
    # セントロイド（に一番近い）サンプルの重複率を調べる
    def decide_choosing_cluster(self, train_df, cluster_df):
        pass

    # サンプリングをする
    def sampling(self):
        pass

    def heuristic(self, train_df, cluster_df):
        int_df = train_df.merge(cluster_df, on="Sample_ID", how="inner")
        cluster_array = np.sort(int_df["Cluster"].unique())
        median_arr = np.array([np.median(int_df[int_df["Cluster"] == c_id]["Score"]) for c_id in cluster_array])
        select_id = np.argsort(median_arr)[:3]
        # threshold = np.mean(median_arr)
        # select_id = np.argwhere(median_arr <= threshold)[:, 0]
        return select_id

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
