import itertools

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
import re
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats


sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa
from library.spectral import SpectralClustering
from library.hierarchical import HierarchicalClustering
from library.visualizer import Visualizer


class Integration:
    def __init__(self, config=None):
        self.config = config

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

    @staticmethod
    def df_filter(df, metric, heuristics):
        df = pd.DataFrame(df[df["Gold"] > 0][[metric, "Term"]])
        df["heuristics"] = heuristics
        return df

    def fitness(self, eval_dir, script_name):
        dir_path = Path(eval_dir)  / script_name
        loss_list = ["analytic", "attention1ep1"]
        finetune_list = ["overlap-attention-{}", "rand-attention-{}"]
        # finetune_list = ["overlap-attention-{}"]
        term_list = ["a", "b", "c", "d"]
        template = "{}_{}_fitness.pkl"
        template_ft = "{}_{}_fitness.finetuning.pkl"
        replace_dict = {"analytic": "MSE", "attention1ep1": "Attention"}

        for data_type in ["train", "test"]:
            fig, axes = plt.subplots(2, 2, figsize=(15, 8))
            for idx, term in enumerate(term_list):
                df_list = []
                for loss_name in loss_list:
                    baseline_df = pd.read_pickle(dir_path / template.format(loss_name, data_type))
                    baseline_df = self.df_filter(baseline_df, metric="Recall_Score", heuristics=loss_name)
                    baseline_df = baseline_df[baseline_df["Term"] == term.upper()]
                    df_list.append(baseline_df)
                for finetune_name in finetune_list:
                    finetune_fixed = finetune_name.format(term)
                    baseline_df = pd.read_pickle(dir_path / template_ft.format(finetune_fixed, data_type))
                    baseline_df = self.df_filter(baseline_df, metric="Recall_Score", heuristics=finetune_fixed)
                    baseline_df = baseline_df[baseline_df["Term"] == term.upper()]
                    df_list.append(baseline_df)

                group_df = pd.concat(df_list)
                term_dict ={"overlap-attention-{}".format(term): "Overlap", "rand-attention-{}".format(term): "Rand"}
                group_df = group_df.replace(dict(**replace_dict, **term_dict))
                ax = axes[idx // 2][idx % 2]
                sns.barplot(data=group_df, x="Term", y="Recall_Score", hue="heuristics", ax=ax)
                ax.set(ylim=(0.4, 1.0))
            os.makedirs(dir_path / "figure", exist_ok=True)
            plt.savefig(dir_path / "figure" / "{}.png".format(data_type))

    def fitness_ft(self, eval_dir, script_name):
        dir_path = Path(eval_dir)  / script_name
        term_list = ["a", "b", "c", "d"]
        heuristics_list = ["overlap", "rand"]

        template = "{}-attention-{}_{}_fitness.finetuning.pkl"

        def print_fitness(metric, data_type="test"):
            df_list = []
            for term, heuristics in itertools.product(term_list, heuristics_list):
                baseline_df = pd.read_pickle(dir_path / template.format(heuristics, term, data_type))
                baseline_df = self.df_filter(baseline_df, metric=metric, heuristics="{}_{}".format(heuristics, term))
                df_list.append(baseline_df)

            group_df = pd.concat(df_list)
            mean = group_df.groupby(["heuristics", "Term"]).mean()
            print("\ntype: {}".format(data_type))
            print(mean)

        import itertools
        type_list = ["train", "test"]
        for _type in type_list:
            print_fitness(metric="Recall_Score", data_type=_type)

    def fitness_tmp(self, eval_dir, script_name):
        dir_path = Path(eval_dir)  / script_name
        # load baseline
        analytic_df = pd.read_pickle(dir_path / "analytic_train_fitness.pkl")
        analytic_df["method"] = "mse"
        attention_df = pd.read_pickle(dir_path / "attention1ep1_train_fitness.pkl")
        attention_df["method"] = "attention"
        # load finetuning
        df_list = []
        for idx in range(3, 9):
            df = pd.read_pickle(dir_path / "finetuning" / "finetuning_train_fitness.A.c10.s{}.pkl".format(idx))
            df["method"] = "selection-{}".format(idx)
            df_list.append(df)
        # integrate
        df_list.extend([analytic_df, attention_df])
        df_list = [df[df["Gold"] > 0.0] for df in df_list]
        df_list = [df[df["Term"] == "A"] for df in df_list]
        integrated_df = pd.concat(df_list)
        print(integrated_df.groupby("method").mean()["Recall_Score"])
        print(integrated_df.groupby("method").std())
        print("")

    def quantitative_random(self, analytic_df, trial_size=10000):
        sample_size_list = [idx for idx in range(20, 101, 20)]
        mean_dict, std_dict = defaultdict(list), defaultdict(list)
        for sample_size in tqdm(sample_size_list):
            for idx in range(trial_size):
                sample_df = analytic_df.sample(n=sample_size)
                mean_dict[sample_size].append(sample_df["Recall_Score"].mean())
                std_dict[sample_size].append(sample_df["Recall_Score"].std())
        # plot
        fig, axes = plt.subplots(2, len(sample_size_list), figsize=(20, 5))
        for idx, sample_size in enumerate(mean_dict.keys()):
            sns.histplot(mean_dict[sample_size], ax=axes[0][idx], binwidth=0.005)
            sns.histplot(std_dict[sample_size], ax=axes[1][idx], binwidth=0.005)
            axes[0][idx].set_xlim(0.55, 0.7)
            axes[1][idx].set_xlim(0.05, 0.20)
            axes[0][idx].set_title("Mean, Sample size = {}".format(sample_size))
            axes[1][idx].set_title("Std, Sample size = {}".format(sample_size))
        plt.tight_layout()
        plt.savefig("tmp/random_fitness.png".format())

    def quantitative_clustering(self, analytic_df, cluster_df, trial_size=10000, cluster_size=10):
        # pass
        selection_size_list = [2, 4, 6, 8, 10]
        mean_dict, std_dict = defaultdict(list), defaultdict(list)
        size_dict = {}
        for i, sample_size in enumerate(selection_size_list):
            print("\nSelection size: {}".format(sample_size))
            sample_df = cluster_df.groupby("Number").apply(lambda x: self.sample(x, n=sample_size)).reset_index(drop=True)
            size_dict[i] = sample_df.groupby("Number")["Number_ID"].count().sum()
            for idx in tqdm(range(trial_size)):
                sample_df = cluster_df.groupby("Number").apply(lambda x: self.sample(x, n=sample_size)).reset_index(drop=True)
                merged_df = analytic_df.merge(right=sample_df, left_on="Sample_ID", right_on="Sample_ID")
                mean_array = merged_df.groupby("Number")["Recall_Score"].mean().to_numpy()
                std_array = merged_df.groupby("Number")["Recall_Score"].std().to_numpy()
                size_array = cluster_df.groupby("Number")["Number_ID"].count().to_numpy()
                size_array = size_array / np.sum(size_array)
                mean_dict[sample_size].append(np.average(mean_array, weights=size_array))
                std_dict[sample_size].append(np.average(std_array, weights=size_array))

        # plot
        fig, axes = plt.subplots(2, len(selection_size_list), figsize=(20, 5))
        for idx, sample_size in enumerate(mean_dict.keys()):
            sns.histplot(mean_dict[sample_size], ax=axes[0][idx], binwidth=0.005)
            sns.histplot(std_dict[sample_size], ax=axes[1][idx], binwidth=0.005)
            axes[0][idx].set_xlim(0.55, 0.7)
            axes[1][idx].set_xlim(0.05, 0.20)
            size = size_dict[idx]
            axes[0][idx].set_title("Mean, Selection size = {}, Sample size = {}".format(sample_size, size))
            axes[1][idx].set_title("Std, Selection size = {}, Sample size = {}".format(sample_size, size))
        plt.tight_layout()
        plt.savefig("tmp/clustering_{}_fitness.png".format(cluster_size))

    def quantitative_fitness(self,  eval_dir, script_name):
        fig, axes = plt.subplots(2, 2, figsize=(14, 7))
        for idx, term in enumerate(["A", "B", "C", "D"]):
            eval_dir_path = Path(eval_dir)  / script_name
            analytic_df = pd.read_pickle(eval_dir_path / "analytic_train_fitness.pkl")
            analytic_df = analytic_df[analytic_df["Term"] == term].reset_index(drop=True)
            analytic_df["Sample_ID"] = [idx for idx in range(len(analytic_df))]
            analytic_df = analytic_df[analytic_df["Gold"] > 0.0]
            ax = axes[idx // 2, idx % 2]
            sns.histplot(analytic_df["Recall_Score"], bins=20, alpha=1.0, binrange=(0.0, 1.0), ax=ax)
            ax.set_title("項目: {}".format(term))
        plt.tight_layout()
        plt.savefig("tmp/{}.png".format(script_name))

    @staticmethod
    def sample(data, n):
        if n >= len(data):
            return data
        else:
            return data.sample(n)


    def __call__(self, prompt_name, eval_dir_path):
        df = self.load_performances(eval_dir_path, prompt_name)
        df = df.sort_values(by='size')
        self.plot(df, prompt_name)


class CheckMasking:
    def __init__(self):
        pass

    def check_masking_efficiency(self, eval_dir, script_name, masking_span, term):
        eval_dir_path = Path(eval_dir) / script_name
        attr_df = pd.read_pickle(eval_dir_path / "analytic_train_attributions.pkl")
        selected_df = attr_df[["Token", "Attribution", "Term"]]
        selected_df = selected_df[selected_df["Term"] == term]
        selected_df["match_idx"] = selected_df["Token"].apply(lambda x: self.find_spans("".join(x[1:-1]), masking_span))
        score_list = []
        text_list = []
        for index, row in selected_df.iterrows():
            if row["match_idx"]:
                score_list.append(np.sum(np.array(row["Attribution"])[row["match_idx"]]))
                text_list.append(np.array(row["Token"])[row["match_idx"]])
        print(np.mean(score_list), np.max(score_list), np.min(score_list))
        print(score_list[:5])
        print(len(attr_df))

    @staticmethod
    def find_spans(text, pattern):
        spans = []
        regex = re.compile(pattern)
        matches = regex.finditer(text)

        for match in matches:
            span = match.span()
            spans.extend([i + 1 for i in range(*span)])

        return spans

