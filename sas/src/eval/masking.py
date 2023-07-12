import torch
import torch.nn.functional as F
import sys
import os
import pandas as pd
import string
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from itertools import islice
import numpy as np
from transformers import BertJapaneseTokenizer
from sklearn.metrics import recall_score, precision_score
from pprint import pprint

sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa
from eval.attribution import FeatureAttribution
from eval.base import EvalBase
from itertools import product
from eval.clustering import Clustering2


class EvalMasking(EvalBase):
    def __init__(self, eval_config, masking_span):
        super().__init__(eval_config)
        self.masking_span = masking_span

    def dump_results(self, dataframe, data_type, suffix, csv=True):
        Util.save_masking_df(dataframe, self.config, data_type, suffix, csv, self.masking_span)

    def execute(self):
        self.model = Util.load_masking_model(self.config, self.model_config, self.masking_span)
        # train set
        train_dataset = Util.load_masking_dataset(self.config, "train", self.masking_span)
        print("Train, size={}".format(len(train_dataset)))
        self.train_size = len(train_dataset)
        self.eval(train_dataset, "train")
        # test set
        pprint(self.config)
        print("Test")
        test_dataset = Util.load_masking_dataset(self.config, "test", self.masking_span)
        self.eval(test_dataset, "test")


class ClusteringMasking(Clustering2):
    def __init__(self, eval_config, masking_span):
        super().__init__(eval_config)
        self.masking_span = masking_span

    def load_attribution_results(self, data_type="train"):
        suffix = "attributions"
        df = Util.load_masking_df(self.config, data_type, suffix, self.masking_span)
        return df

    def clustering(self, df, data_type):
        term_list, score_list = df["Term"].unique(), df["Pred"].unique()
        for term in term_list:
            part_df = df[(df["Pred"] != 0) & (df["Term"] == term)]
            print("{}, Term: {}".format(self.config.script_name, term))
            data_points = self.select_attribution(part_df, self.config.point_type)
            # make hierarchy
            hierarchy = self.make_hierarchy(data_points)
            score, score_list = self.masking_span, part_df["Pred"].to_list()
            # dump data
            data_df = self.integrate_df(part_df, term)
            self.dump_data_df(data_df, data_type, term, score)
            # make clustering
            attributions, sample_id = part_df["Attribution"], part_df["Sample_ID"]
            cluster_df_list = self.fcluster(hierarchy, attributions, sample_id, data_points)
            self.plot_dendrogram(hierarchy, cluster_df_list, data_type, term, score)
            self.dump_cluster_df(cluster_df_list, data_type, term, score)

    def make_clustering_results(self):
        print("Train set")
        train_df = self.load_attribution_results(data_type="train")
        self.clustering(train_df, data_type="train")
        print("Test set")
        test_df = self.load_attribution_results(data_type="test")
        self.clustering(test_df, data_type="test")
