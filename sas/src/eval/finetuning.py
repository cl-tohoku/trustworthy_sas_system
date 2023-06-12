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


class EvalFinetuning(EvalBase):
    def __init__(self, eval_config):
        super().__init__(eval_config)
        self.term, self.cluster_size, self.selection_size = None, None, None

    def dump_finetuning_results(self, dataframe, data_type, suffix, csv):
        Util.save_finetuning_df(dataframe, self.config, data_type, suffix, csv, self.term, self.cluster_size, self.selection_size)

    def finetuning_eval(self, dataset, data_type):
        self.model.eval()

        results = self.eval_scripts(dataset)
        performances = self.eval_performance(results)

        dataframe = pd.DataFrame(results)
        dataframe_performance = pd.DataFrame(performances)
        self.dump_finetuning_results(dataframe, suffix="results", data_type=data_type, csv=True)
        self.dump_finetuning_results(dataframe_performance, suffix="performances", data_type=data_type, csv=True)

        if self.config.attribution:
            attribution_df, fitness_df = self.eval_attributions(dataset)
            print("Outputting...")
            self.dump_finetuning_results(attribution_df, suffix="attributions", data_type=data_type, csv=False)
            self.dump_finetuning_results(fitness_df, suffix="fitness", data_type=data_type, csv=True)

    def evaluate(self):
        self.model = Util.load_finetuning_model(self.config, self.model_config, self.term, self.cluster_size, self.selection_size)
        # test set
        print("Test")
        test_dataset = Util.load_dataset(self.config, "test",)
        self.finetuning_eval(test_dataset, "test")
        # train set
        print("Train")
        train_dataset = Util.load_dataset(self.config, "train",)
        self.finetuning_eval(train_dataset, "train")
        print("Finetuning")
        finetuning_dataset = Util.load_finetuning_dataset(self.config, "train", self.term, self.cluster_size, self.selection_size)

    def execute(self, given_term="A"):
        # term_list = list(string.ascii_uppercase)[:self.model_config.output_size][:1]
        cluster_list, selection_list = list(range(10, 11)), list(range(3, 11))
        for cluster_size, selection_size in product(cluster_list, selection_list):
            print("term: {}, c_size: {}, s_size: {}".format(given_term, cluster_size, selection_size))
            self.term, self.cluster_size, self.selection_size = given_term, cluster_size, selection_size
            self.evaluate()
