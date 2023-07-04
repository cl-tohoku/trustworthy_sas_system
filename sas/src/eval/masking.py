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

    def dump_finetuning_results(self, dataframe, data_type, suffix, csv):
        Util.save_masking_df(dataframe, self.config, data_type, suffix, csv, self.masking_span)

    def execute(self):
        self.model = Util.load_masking_model(self.config, self.model_config, self.masking_span)
        # test set
        pprint(self.config)
        print("Test")
        test_dataset = Util.load_masking_dataset(self.config, "test", self.masking_span)
        self.eval(test_dataset, "test")
        # train set
        print("Train")
        train_dataset = Util.load_masking_dataset(self.config, "train", self.masking_span)
        self.eval(train_dataset, "train")


class ClusteringMasking(Clustering2):
    def __init__(self, eval_config, masking_span):
        super().__init__(eval_config)
        self.masking_span = masking_span
