import pickle
import sys
import os
import torch
from glob import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering, KMeans
from collections import defaultdict
import seaborn as sns
import Levenshtein
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import kl_div, rel_entr
from sklearn.metrics import  recall_score


sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa
# from library.spectral import SpectralClustering
from library.hierarchical import HierarchicalClustering
from library.visualizer import Visualizer
from library.dbscan import DBSCANAnalyzer
from library.selector import Selector


class Clustering2:
    def __init__(self, config_path):
        self.config = config_path
        self.prompt_config = Util.load_prompt(self.config)
        self.selection_size = 5

    def load_attribution_results(self, data_type="train"):
        suffix = "attributions"
        df = Util.load_eval_df(self.config, data_type, suffix)
        return df

    def select_attribution(self, df, point_type):
        if point_type == "embedding":
            attributions = Selector.mean(df["Embedding"].to_list())
        elif point_type == "multiple-norm":
            attributions = Selector.emb_dot_attr_norm(df["Integrated_Gradients"].to_list(), df["Embedding"].to_list())
        elif point_type == "attribution-only":
            attributions = Selector.mean(df["Integrated_Gradients"].to_list())
        elif point_type == "multiple-attr":
            attributions = Selector.emb_dot_attr(df["Integrated_Gradients"].to_list(), df["Embedding"].to_list())
        elif point_type == "long-tail":
            attributions = Selector.long_tail(df["Integrated_Gradients"].to_list())
        elif point_type == "counter":
            attributions = Selector.counter(df["Attribution"].to_list(), df["Token"].to_list())
        else:
            raise RuntimeError

        return attributions

    def make_cluster_df(self, attributions, cluster_size):
        # Hierarchical
        clustering_instance = HierarchicalClustering()
        hierarchy = clustering_instance.linkage(attributions)
        hc_label = clustering_instance.fcluster(t=cluster_size)
        return hierarchy, hc_label,

    def integrate_df(self, df, term):
        heatmap = df["Attribution"].to_list()
        token = df["Token"].to_list()
        annotation = df["Annotation"].to_list()
        colormap = Visualizer.attribution_to_color(heatmap)
        masked_colormap = Visualizer.attribution_to_color(heatmap, annotation)
        idx_list = [idx for idx in range(len(heatmap))]
        # integrate
        integrated_df = pd.DataFrame({"Idx": idx_list, "Token": token, "Color": colormap,
                                      "Mask_Color": masked_colormap, "Annotation": annotation})
        integrated_df["Attribution"] = heatmap
        integrated_df["Pred"] = df["Pred"].to_list()
        integrated_df["Gold"] = df["Gold"].to_list()
        integrated_df["Sample_ID"] = df["Sample_ID"].to_list()
        integrated_df["Term"] = term
        integrated_df["Preprocessing"] = self.config.preprocessing_type
        return integrated_df


    def load_data_dict(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def make_output_image_name(self, term, category, image_name, suffix=".png"):
        return "{}_{}_{}_{}.{}".format(term, category, self.config.point_type, image_name, suffix)

    def plot_tsne(self, attributions, label, score_list=None, output_path=None):
        figsize = (7, 5)
        figure_path = output_path / "tsne.png"
        Visualizer.tsne(cluster_id=label, matrix=attributions, output_path=figure_path,
                        score_list=score_list, figsize=figsize)

    def save_dendrogram_range(self, hierarchy, attribution, sample_id, vector, output_path,):
        cluster_df_list = []
        for k in tqdm(range(2, 31)):
            figsize = (2, k)
            k_path = output_path / str(k)
            os.makedirs(k_path, exist_ok=True)
            figure_path = k_path / "dendrogram.png"
            # Sample ID -> table of correspondences, Number ID -> cluster index
            cluster_df = Visualizer.dendrogram(hierarchy=hierarchy, output_path=figure_path, figsize=figsize, k=k)
            attr_df = pd.DataFrame({"Attribution": attribution.tolist(), "Vector": vector.tolist(),
                                    "Sample_ID": sample_id.to_list()})
            cluster_df = cluster_df.merge(right=attr_df, left_on="Number_ID", right_index=True).sort_index()
            df_path = k_path / "cluster.pkl"
            cluster_df.to_pickle(df_path)
            cluster_df_list.append(cluster_df)
        return cluster_df_list

    def make_clustering_results(self):
        def clustering(df, data_type):
            term_list, score_list = df["Term"].unique(), df["Pred"].unique()
            for term in term_list:
                part_df = df[(df["Pred"] != 0) & (df["Term"] == term)]
                print("{}, Term: {}".format(self.config.script_name, term))
                attributions = self.select_attribution(part_df, self.config.point_type)
                hierarchy, hc_id = self.make_cluster_df(attributions, cluster_size=10)
                score, score_list = "R", part_df["Pred"].to_list()
                # dump data
                setting_dir = "{}_{}_{}".format(self.config.script_name, term, score)
                output_dir = Path(self.config.cluster_dir) / data_type / setting_dir
                os.makedirs(output_dir, exist_ok=True)
                data_df = self.integrate_df(part_df, term)
                data_df.to_pickle(str(output_dir / "data.xz.pkl"), compression="xz")
                # plot t-sne
                self.plot_tsne(attributions, label=hc_id, score_list=score_list, output_path=output_dir)
                # plot dendrogram & save clustering ID
                _ = self.save_dendrogram_range(hierarchy, attribution=part_df["Attribution"], sample_id=part_df["Sample_ID"],
                                               vector=attributions, output_path=output_dir)
                # select for heuristic
                # merge_df = self.heuristic_selector(data_df, cluster_df_list[self.selection_size - 1], attributions)
                # os.makedirs(Path(self.config.finetuning_dir) / data_type / setting_dir, exist_ok=True)
                # merge_df.to_pickle(Path(self.config.finetuning_dir) / data_type / setting_dir / "heuristic.xz.pkl", compression="xz")

        print("Train set")
        train_df = self.load_attribution_results(data_type="train")
        clustering(train_df, data_type="train")
        print("Test set")
        test_df = self.load_attribution_results(data_type="test")
        clustering(test_df, data_type="test")
