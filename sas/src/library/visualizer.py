import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
# import japanize_matplotlib
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

sys.path.append("..")
from library.util import Util
from library.quadratic_weighted_kappa import quadratic_weighted_kappa

import matplotlib.font_manager
matplotlib.font_manager.fontManager.addfont("/ipaexg.ttf")
# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])
sns.set_theme(style="whitegrid", font="IPAexGothic")
marker_list = ["o", "v", "s", "p", "P", "*", "D", "d", "h", "H"]

class Visualizer:
    @staticmethod
    def generate_html(tokens, color_info, formatter, params):
        html_start, html_end = "<html><body>", "</body></html>"

        columns = params.columns
        for t_idx, (token, c_info) in enumerate(zip(tokens, color_info)):
            text = ""
            for c in columns:
                series = params[c]
                text += "{}={}, ".format(c, series.iloc[t_idx])
            text += "Text="
            for c_idx, char in enumerate(token):
                text += formatter(char, c_info, c_idx)
            text += "<br>"
            html_start += text

        html = html_start + html_end
        return html

    @staticmethod
    def attribution_to_color(attributions, annotations=None):
        color_list = []
        # max_value = np.percentile([np.max(a) for a in attributions], q=95)
        for a_idx, attr in enumerate(attributions):
            # scaling
            scaled_attr = np.array(attr) / np.linalg.norm(attr)
            scaled_attr *= 128
            scaled_attr = np.floor(scaled_attr) + 128
            scaled_attr[scaled_attr > 255.0] = 255.0
            scaled_attr[scaled_attr < 0.0] = 0.0
            scaled_attr[scaled_attr < 128.0] = 128.0

            # attr -> rgb -> colorcode
            scaled_list = []
            for s_idx, scaled in enumerate(scaled_attr):
                rgb = plt.cm.bwr(int(scaled), bytes=True)
                if annotations is None:
                    scaled_list.append('#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2])))
                elif annotations[a_idx][s_idx] == 1:
                    scaled_list.append('#4169e1')
                else:
                    scaled_list.append('#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2])))

            color_list.append(scaled_list)
        return color_list

    @staticmethod
    def heatmap_simple(tokens, rankings, params, output_path="heatmap.html"):
        def formatter(char, c_info, c_idx):
            if c_idx in c_info:
                return "<mark>{}</mark>".format(char)
            else:
                return char

        html = Visualizer.generate_html(tokens, rankings, formatter=formatter, params=params)
        with open(output_path, "w") as f:
            f.write(html)

    @staticmethod
    def heatmap_rich(tokens, attributions, params, annotations=None, output_path=Path("heatmap.html")):
        # get color
        color_list = Visualizer.attribution_to_color(attributions, annotations)

        def formatter(char, c_info, c_idx):
            return '<span style="background-color:{}">{}</span>'.format(c_info[c_idx], char)

        html = Visualizer.generate_html(tokens, color_list, formatter=formatter, params=params)
        with open(output_path, "w") as f:
            f.write(html)


    @staticmethod
    def eigen_value(eigen_value, output_path="eigen.png", xlim=10):
        figsize = (5, 3)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # plot eigen value
        value_df = pd.DataFrame({"N": np.arange(len(eigen_value)), "Eigen value": eigen_value})
        sns.pointplot(value_df, x="N", y="Eigen value", ax=ax)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("i番目の固有値（昇順）")
        ax.set_ylabel("固有値")

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close()

    @staticmethod
    def cluster_size(counter, output_path="eigen.png"):
        figsize = (7, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # plot eigen value
        cluster = [str(c) for c in counter.index]
        value_df = pd.DataFrame({"Cluster Number": cluster, "Count": counter})
        value_df = value_df.sort_values(by="Count", ascending=False).reset_index(drop=True)
        sns.barplot(value_df, x="Cluster Number", y="Count", ax=ax)
        # ax.set_xlim(0, 100)
        ax.set_xlabel("クラスタID")
        ax.set_ylabel("件数")

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close()

    @staticmethod
    def tsne(cluster_df, output_path="cluster.png", figsize=(7, 4)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
        sorted_df = cluster_df.sort_values("Number_ID").reset_index(drop=True)
        points = np.array(sorted_df["Point"].to_list())
        embedded = tsne.fit_transform(points)
        vector_df = pd.DataFrame({"X": embedded[:, 0], "Y": embedded[:, 1], "Cluster": sorted_df["Number"]})
        sns.scatterplot(vector_df, x="X", y="Y", hue="Cluster", alpha=1.0, palette='colorblind', ax=ax)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("t-SNE")

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close()

    @staticmethod
    def dendrogram(hierarchy, k, output_path, figsize=(12, 5),):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        dendrogram(hierarchy, truncate_mode='lastp', p=k, orientation='right', ax=ax)
        ax.set_title("Dendrogram")
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close()

    @staticmethod
    def sse_elbow(sse_df, output_path):
        fig, ax = plt.subplots(1, 1,)
        sns.lineplot(data=sse_df, x="Cluster_Size", y="SSE", ax=ax)
        plt.savefig(output_path)

    @staticmethod
    def score_hist(df, output_path):
        fig, ax = plt.subplots(1, 1,)
        sns.histplot(data=df, x="Score", bins=20, ax=ax)
        ax.set_xlim(-1.0, 1.0)
        plt.savefig(output_path)
