import torch
import torch.nn.functional as F
import sys
import os
from glob import glob
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa
from library.spectral import SpectralClustering
from library.hierarchical import HierarchicalClustering
from library.visualizer import Visualizer


class Integration:
    def __init__(self):
        pass

    def load_performances(self, eval_dir_path, prompt_name):
        dir_list = glob(eval_dir_path + "/*")
        df_list = []
        for dir_path in dir_list:
            if prompt_name in dir_path:
                df = pd.read_pickle(Path(dir_path) / "analytic_test_performances.pkl")
                size = dir_path.replace(str(Path(eval_dir_path) / prompt_name), "")
                if size == "":
                    continue
                size = int(size[1:])
                df["size"] = size
                df_list.append(df)
        return pd.concat(df_list).reset_index(drop=True)

    def plot(self, df, prompt_name):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.lineplot(data=df, x="size", y="RMSE", hue="item", ax=axes[0], style="item", markers=True, dashes=False)
        sns.lineplot(data=df, x="size", y="QWK", hue="item", ax=axes[1], style="item", markers=True, dashes=False)
        dir_path = "data/integration"
        os.makedirs(dir_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(Path(dir_path) / "{}.png".format(prompt_name))

    def __call__(self, prompt_name, eval_dir_path):
        df = self.load_performances(eval_dir_path, prompt_name)
        df = df.sort_values(by='size')
        self.plot(df, prompt_name)
