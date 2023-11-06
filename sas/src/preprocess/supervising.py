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
        cluster_df_dir = Path(self.config.cluster_dir) / "train" / self.config.prev_script_name / "cluster"
        file_name = "{}_{}_{}.gzip.pkl".format(self.config.sf_term, self.config.target_score, cluster_k)
        cluster_df_path = cluster_df_dir / file_name
        cluster_df = pd.read_pickle(cluster_df_path, compression="gzip")
        return cluster_df

    # クラスタ数を決定する
    # SSE が一定のしきい値以下になる最小のクラスタ数を推定
    def decide_cluster_df(self):
        minimum_size = 5
        cluster_range = (minimum_size, 11)
        inertia_list = []
        for cluster_k in range(*cluster_range):
            cluster_df = self.load_cluster(cluster_k)
            data_points = np.array([np.array(dp) for dp in cluster_df["Data_Point"]])
            cluster_labels = cluster_df["Cluster"].to_numpy()
            inertia_list.append(np.mean(np.min(1 - cosine_similarity(data_points, np.array([np.mean(data_points[cluster_labels == i], axis=0) for i in range(cluster_k)])), axis=1)))
        arg = np.argwhere(np.array(inertia_list) < self.config.threshold)[:, 0]
        desired_cluster_k = np.min(arg) + minimum_size if arg.size != 0 else 10
        return self.load_cluster(desired_cluster_k)

    # 選ぶクラスタを（ヒューリスティクスに）決定する
    # セントロイド（に一番近い）サンプルの重複率を調べる
    # 良い / 駄目の基準は、最大の importance がルーブリック外に割り当てられているか
    def calculate_centroid(self, data_points):
        sim_matrix = cosine_similarity(data_points)
        similarity_sum = np.sum(sim_matrix, axis=1)
        centroid_index = np.argmax(similarity_sum)
        return centroid_index

    def check_attribution(self, attribution, annotation):
        return annotation[np.argmax(attribution)] == 0

    def decide_choosing_cluster(self, train_df, cluster_df):
        int_df = train_df.merge(cluster_df, on="Sample_ID", how="inner")
        chosen_cluster = []
        for cluster_number, group in int_df.groupby("Cluster"):
            chosen_list, data_points = group["Chosen"].tolist(), group["Data_Point"].tolist()
            centroid_idx = self.calculate_centroid(data_points)
            if chosen_list[centroid_idx]:
                chosen_cluster.append(cluster_number)

        return chosen_cluster

    # サンプリングをする
    def sampling(self, train_df, cluster_df, chosen_list):
        int_df = train_df.merge(cluster_df, on="Sample_ID", how="inner")
        chosen_df = int_df[int_df["Cluster"].isin(chosen_list)]
        sampling_size = self.config.sampling_size
        chosen_df.groupby("Cluster").apply(lambda x: x.sample(n=sampling_size, replace=False) if len(x) > sampling_size else x)
        return chosen_df.reset_index(drop=True)

    # method
    def execute(self):
        # load dataset
        prep_name = self.config.preprocess_name
        dataset_dir = self.config.dataset_dir
        script_name = self.config.script_name
        prev_mode = self.config.prev_mode

        if "superficial" in prev_mode.lower():
            sf_term, sf_idx = self.config.sf_term, self.config.sf_idx
            train_df = Util.load_sf_dataset(sf_term, sf_idx, prep_name, "train", dataset_dir)
            valid_df = Util.load_sf_dataset(sf_term, sf_idx, prep_name, "valid", dataset_dir)
            test_df = Util.load_sf_dataset(sf_term, sf_idx, prep_name, "test", dataset_dir)
        else:
            train_df = Util.load_dataset_static(prep_name, "train", prev_mode, dataset_dir)
            valid_df = Util.load_dataset_static(prep_name, "valid", prev_mode, dataset_dir)
            test_df = Util.load_dataset_static(prep_name, "test", prev_mode, dataset_dir)

        cluster_df = self.decide_cluster_df()
        chosen_list = self.decide_choosing_cluster(train_df, cluster_df)
        if not chosen_list:
            return 1

        sampled_df = self.sampling(train_df, cluster_df, chosen_list)

        # output
        self.to_pickle(sampled_df, "chosen", script_name)

        # output default data
        self.to_pickle(train_df, "train", script_name)
        self.to_pickle(valid_df, "valid", script_name)
        self.to_pickle(test_df, "test", script_name)

        # load dataset & parse
        prompt = Util.load_prompt_config(self.config.prompt_path)
        self.dump_prompt(prompt)
        return 0


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
