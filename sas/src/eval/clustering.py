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
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import kl_div, rel_entr
from sklearn.metrics import  recall_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

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

    def make_hierarchy(self, attributions):
        # Hierarchical
        clustering_instance = HierarchicalClustering()
        hierarchy = clustering_instance.linkage(attributions)
        return hierarchy

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

    def dump_data_df(self, data_df, data_type, term, score):
        setting_dir = "{}_{}_{}".format(self.config.script_name, term, score)
        output_dir = Path(self.config.cluster_dir) / data_type / setting_dir
        os.makedirs(output_dir, exist_ok=True)
        data_df.to_pickle(str(output_dir / "data.xz.pkl"), compression="xz")

    def fcluster(self, hierarchy, attribution, sample_id, data_points, t_range=(2, 31)):
        cluster_df_list = []
        for k in tqdm(range(*t_range)):
            attr_df = pd.DataFrame({"Attribution": attribution.tolist(), "Point": data_points.tolist(),
                                    "Sample_ID": sample_id.to_list()})

            cluster_id = fcluster(hierarchy, t=k, criterion='maxclust')
            sort_id = [int(i) for i in dendrogram(hierarchy, no_plot=True)['ivl']]
            cluster_id = (cluster_id - k - 1) * (-1)
            idx_list = [idx for idx in range(len(cluster_id))]
            df = pd.DataFrame({"Number": cluster_id, "Number_ID": idx_list})
            df = df.iloc[sort_id]

            cluster_df = df.merge(right=attr_df, left_on="Number_ID", right_index=True).sort_index()
            cluster_df_list.append(cluster_df)
        return cluster_df_list

    def plot_dendrogram(self, hierarchy, cluster_df_list, data_type, term, score, t_range=(2, 31)):
        for k in tqdm(range(*t_range)):
            cluster_df = cluster_df_list[k - t_range[0]]
            setting_dir = "{}_{}_{}".format(self.config.script_name, term, score)
            output_dir = Path(self.config.cluster_dir) / data_type / setting_dir
            k_path = output_dir / str(k)
            os.makedirs(k_path, exist_ok=True)
            dendro_path = k_path / "dendrogram.png"
            Visualizer.dendrogram(hierarchy, k, output_path=dendro_path, figsize=(2, k))
            tsne_path = k_path / "tsne.png"
            Visualizer.tsne(cluster_df, output_path=tsne_path)

    def dump_cluster_df(self, cluster_df_list, data_type, term, score, t_range=(2, 31)):
        for k in range(*t_range):
            setting_dir = "{}_{}_{}".format(self.config.script_name, term, score)
            output_dir = Path(self.config.cluster_dir) / data_type / setting_dir
            k_path = output_dir / str(k)
            os.makedirs(k_path, exist_ok=True)
            df_path = k_path / "cluster.pkl"
            cluster_df_list[k - t_range[0]].to_pickle(str(df_path))

    def clustering(self, df, data_type):
        term_list, score_list = df["Term"].unique(), df["Pred"].unique()
        for term in term_list:
            part_df = df[(df["Pred"] != 0) & (df["Term"] == term)]
            print("{}, Term: {}".format(self.config.script_name, term))
            data_points = self.select_attribution(part_df, self.config.point_type)
            # make hierarchy
            hierarchy = self.make_hierarchy(data_points)
            score, score_list = "R", part_df["Pred"].to_list()
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
