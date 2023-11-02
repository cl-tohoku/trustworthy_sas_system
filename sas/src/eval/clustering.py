import pickle
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import seaborn as sns
import itertools

sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa
# from library.spectral import SpectralClustering
from library.hierarchical import HierarchicalClustering
from library.visualizer import Visualizer
from library.dbscan import DBSCANAnalyzer
from library.selector import Selector


class Clustering:
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
        inertia = np.mean(np.min(1 - cosine_similarity(data_points, np.array([np.mean(data_points[cluster_labels == i], axis=0)
                                                                              for i in range(cluster_k)])), axis=1))
        return cluster_labels.tolist(), inertia

    def hierarchical(self, data_points, cluster_k):
        tmp = np.sum(data_points, axis=1)
        cosine_distance = 1.0 - cosine_similarity(data_points)
        Z = linkage(cosine_distance, method='ward')
        cluster_labels = fcluster(Z, t=cluster_k, criterion='maxclust') - 1
        cluster_labels = np.max(cluster_labels) - cluster_labels
        inertia = np.mean(np.min(1 - cosine_similarity(data_points, np.array([np.mean(data_points[cluster_labels == i], axis=0)
                                                                              for i in range(cluster_k)])), axis=1))
        return cluster_labels.tolist(), inertia, Z

    def plot_dendrogram(self, Z, cluster_labels, cluster_k, output_path):
        plt.figure(figsize=(2, max(cluster_labels) + 1))
        dendro = dendrogram(Z, labels=cluster_labels, truncate_mode='lastp', p=cluster_k,
                            orientation='left')
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_scatter(self, data_points, labels, output_path):
        # do tsne
        cosine_distances = data_points
        compressor = KernelPCA(n_components=2, random_state=0)
        results = compressor.fit_transform(cosine_distances)

        # plot figure        
        plt.figure(figsize=(7, 4))
        for cluster in np.unique(labels):
            plt.scatter(results[labels == cluster, 0], results[labels == cluster, 1],
                        label=f"Cluster {cluster}", alpha=0.5)
        plt.legend()
        plt.title("Clustering with visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(output_path)

    def calc_score(self, part_df):
        annotation, attribution = part_df["Annotation"], part_df["Attribution"]
        score_list = []
        for anno, attr in zip(annotation, attribution):
            attr = np.array(attr)
            attr[attr < 0.0] = 0.0
            attr_all = np.sum(attr)
            attr_lap = np.dot(attr, anno)
            if attr_lap <= 0.0:
                score_list.append(-1.0)
            else:
                score_list.append(attr_lap / attr_all)
        score_array = np.array(score_list)
        return list(score_array[score_array != -1.0])

    def plot_score(self, data_df, cluster_df, output_path):
        int_df = data_df.merge(cluster_df, on="Sample_ID", how="inner")
        cluster_array = np.sort(int_df["Cluster"].unique())
        score_list, id_list = [], []
        for cluster_id in cluster_array:
            part_df = int_df[int_df["Cluster"] == cluster_id]
            s_list = self.calc_score(part_df)
            score_list.extend(s_list)
            id_list.extend([cluster_id for _ in s_list])

        result_df = pd.DataFrame({"Score": score_list, "Cluster": id_list})
        plt.figure()
        sns.boxplot(data=result_df, y="Score", x="Cluster")
        plt.xlabel("Cluster ID")
        plt.ylabel("Ratio score")
        plt.tight_layout()
        plt.savefig(output_path)

    def calc_score_2(self, part_df):
        def calc_ratio_score(anno, attr):
            attr = np.array(attr)
            attr[attr < 0.0] = 0.0
            attr_all, attr_lap = np.sum(attr), np.dot(attr, anno)
            return -1.0 if attr_lap <= 0.0 else attr_lap / attr_all

        score_list = part_df.apply(lambda x: calc_ratio_score(x["Annotation"], x["Attribution"]), axis=1).tolist()
        return score_list

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
        os.makedirs(output_dir / "cluster", exist_ok=True)
        os.makedirs(output_dir / "dendrogram", exist_ok=True)
        os.makedirs(output_dir / "scatter", exist_ok=True)
        os.makedirs(output_dir / "inertia", exist_ok=True)
        os.makedirs(output_dir / "score", exist_ok=True)
        data_df.to_pickle(output_dir / "data_df.gzip.pkl", compression="gzip")

        # clustering setting
        cluster_k_list = list(range(2, 21))
        term_list = df["Term"].unique().tolist()

        for term in term_list:
            sliced_df = df[df["Term"] == term]
            score_list = sliced_df["Pred"].unique().tolist()
            for score in score_list:
                sliced_2_df = sliced_df[sliced_df["Pred"] == score]
                data_points = self.transform_attribution(sliced_2_df)
                inertia_list = []
                if len(sliced_2_df) < 30:
                    continue
                if score == 0:
                    continue
                for cluster_k in tqdm(cluster_k_list):
                    # calculate cosine similarity for each data points
                    # clustering
                    labels, inertia, Z = self.hierarchical(data_points=data_points, cluster_k=cluster_k)
                    # make data df & output
                    cluster_dict = dict()
                    cluster_dict["Sample_ID"] = sliced_2_df["Sample_ID"].to_list()
                    cluster_dict["Cluster"] = labels
                    # calc score
                    score_list = self.calc_score_2(sliced_2_df)
                    cluster_dict["Score"] = score_list
                    # output data
                    cluster_df = pd.DataFrame(cluster_dict)
                    file_name = "{}_{}_{}.gzip.pkl".format(term, score, cluster_k)
                    cluster_df.to_pickle(output_dir / "cluster" / file_name, compression="gzip")
                    # plot dendro
                    file_name = "{}_{}_{}.png".format(term, score, cluster_k)
                    self.plot_dendrogram(Z, labels, cluster_k, output_dir / "dendrogram" / file_name)
                    # plot scatter
                    file_name = "{}_{}_{}.png".format(term, score, cluster_k)
                    self.plot_scatter(data_points, labels, output_dir / "scatter" / file_name)
                    inertia_list.append(inertia)
                    # plot score
                    self.plot_score(sliced_2_df, cluster_df, output_dir / "score" / file_name)

                # plot inertia
                file_name = "{}_{}.png".format(term, score)
                self.plot_inertia(cluster_k_list, inertia_list, output_dir / "inertia" / file_name)

    def make_clustering_datasets(self):
        print("Train set")
        train_df = self.load_attribution_results(data_type="train")
        self.process(train_df, data_type="train")

        print("Test set")
        test_df = self.load_attribution_results(data_type="test")
        self.process(test_df, data_type="test")
