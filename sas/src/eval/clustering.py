import pickle
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances
import hdbscan
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import itertools

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


class ForClustering:
    def __init__(self, config_path):
        self.config = config_path
        self.prompt_config = Util.load_prompt(self.config)
        self.attribution_name = "Attribution"

    def load_attribution_results(self, data_type="train"):
        suffix = "attributions"
        df = Util.load_eval_df(self.config, data_type, suffix)
        return df

    def transform_attribution(self, df):
        return Selector.counter(df[self.attribution_name].to_list(), df["Token"].to_list())

    def hdbscan(self, cosine_distance):
        cluster_clf = hdbscan.HDBSCAN(min_cluster_size=3)
        labels = cluster_clf.fit_predict(cosine_distance)
        return labels.tolist()

    def spectrum(self, data_points, cluster_k):
        clustering = SpectralClustering(n_clusters=cluster_k, affinity="rbf", random_state=42, n_jobs=-1)
        cluster_labels = clustering.fit_predict(data_points)

        # イナーシャの計算 (コサイン類似度なので、1からの差を考慮)
        inertia = sum(np.min(1 - cosine_similarity(data_points, np.array([np.mean(data_points[cluster_labels == i], axis=0) for i in range(cluster_k)])), axis=1))
        return cluster_labels.tolist(), inertia

    def hierarchical(self, data_points, cluster_k):
        clustering = AgglomerativeClustering(n_clusters=cluster_k, affinity='euclidean', linkage='ward')
        clustering.fit(data_points)
        cluster_labels = clustering.labels_
        inertia = sum(np.min(1 - cosine_similarity(data_points, np.array([np.mean(data_points[cluster_labels == i], axis=0) for i in range(cluster_k)])), axis=1))
        return cluster_labels.tolist(), inertia

    def plot_scatter(self, data_points, labels, output_path):
        # do tsne
        cosine_distances = data_points
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(cosine_distances)        

        # plot figure        
        plt.figure(figsize=(7, 4))
        for cluster in np.unique(labels):
            plt.scatter(tsne_results[labels == cluster, 0], tsne_results[labels == cluster, 1],
                        label=f"Cluster {cluster}", alpha=0.5)
        plt.legend()
        plt.title("Clustering with T-SNE visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_inertia(self, cluster_k_list, inertia_list, output_path):
        plt.figure(figsize=(7, 4))
        plt.plot(cluster_k_list, inertia_list, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method')
        plt.tight_layout()
        plt.savefig(output_path)

    def process(self, df, data_type):
        # generate color map
        colormap = Visualizer.attribution_to_color(df[self.attribution_name])
        masked_colormap = Visualizer.attribution_to_color(df[self.attribution_name], df["Annotation"])

        # make data df
        data_dict = dict()
        data_dict["Sample_ID"], data_dict["Term"] = df["Sample_ID"].to_list(), df["Term"].to_list()
        data_dict["Color"] = colormap
        data_dict["Masked_Color"] = masked_colormap
        data_dict["Token"], data_dict["Annotation"] = df["Token"].to_list(), df["Annotation"].to_list()
        data_dict["Pred"], data_dict["Gold"] = df["Pred"].to_list(), df["Gold"].to_list()

        # output data df
        data_df = pd.DataFrame(data_dict)
        output_dir = Path(self.config.cluster_dir) / data_type / self.config.script_name
        os.makedirs(output_dir, exist_ok=True)
        data_df.to_pickle(output_dir / "data_df.gzip.pkl", compression="gzip")

        # clustering setting
        cluster_k_list = list(range(2, 21))
        term_list = df["Term"].unique().tolist()

        for term in tqdm(term_list):
            sliced_df = df[df["Term"] == term]
            score_list = sliced_df["Pred"].unique().tolist()
            for score in score_list:
                sliced_2_df = sliced_df[sliced_df["Pred"] == score]
                data_points = self.transform_attribution(sliced_2_df)
                inertia_list = []
                if len(sliced_2_df) < 21:
                    continue
                for cluster_k in cluster_k_list:
                    # calculate cosine similarity for each data points
                    # clustering
                    # labels, inertia = self.spectrum(data_points=data_points, cluster_k=cluster_k)
                    labels, inertia = self.hierarchical(data_points=data_points, cluster_k=cluster_k)
                    # make data df & output
                    cluster_dict = dict()
                    cluster_dict["Sample_ID"] = sliced_2_df["Sample_ID"].to_list()
                    cluster_dict["Cluster"] = labels
                    # output data
                    cluster_df = pd.DataFrame(cluster_dict)
                    file_name = "cluster_df_{}_{}_{}.gzip.pkl".format(term, score, cluster_k)
                    cluster_df.to_pickle(output_dir / file_name, compression="gzip")
                    # plot scatter
                    file_name = "scatter_{}_{}_{}.png".format(term, score, cluster_k)
                    self.plot_scatter(data_points, labels, output_dir / file_name)
                    inertia_list.append(inertia)

                # plot inertia
                file_name = "inertia_{}_{}".format(term, score)
                self.plot_inertia(cluster_k_list, inertia_list, output_dir / file_name)

    def make_clustering_datasets(self):
        print("Train set")
        train_df = self.load_attribution_results(data_type="train")
        self.process(train_df, data_type="train")

        print("Test set")
        test_df = self.load_attribution_results(data_type="test")
        self.process(test_df, data_type="test")

