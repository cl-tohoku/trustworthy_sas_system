import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer



"""
kmeans = KMeans(n_clusters=2).fit(np.expand_dims(norm_list, axis=1))
labels = kmeans.labels_
center = kmeans.cluster_centers_
print(len(norm_list))
print(center)
hist, bins = np.histogram(norm_list, bins=50)
logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
plt.clf()
norm_array = np.array(norm_list)
plt.hist(x=norm_array[labels==0], bins=logbins, label=0)
plt.hist(x=norm_array[labels==1], bins=logbins, label=1)
plt.xscale('log')
plt.legend()
plt.savefig("tmp/hist_norm_{}.png".format(len(norm_list)))
"""


class Selector:
    def __init__(self):
        pass

    @staticmethod
    def mean(attr_obj):
        return np.array([np.mean(obj, axis=0) for obj in attr_obj])

    @staticmethod
    def long_tail(attr_obj, k=5):
        temporary_list = []
        for obj in attr_obj:
            norm_array = np.linalg.norm(obj, axis=1)
            top_k_arg = np.argsort(norm_array)[::-1][:int(np.ceil(len(norm_array) * 0.2))]
            mean_array = np.mean(np.array(obj)[top_k_arg], axis=0)
            temporary_list.append(mean_array)
        return np.array(temporary_list)

    @staticmethod
    def emb_dot_attr_norm(attr_obj, emb_obj):
        temporary_list = []
        norm_list = []
        for attr, emb in zip(attr_obj, emb_obj):
            # attr_norm = np.expand_dims(np.linalg.norm(attr, axis=1), axis=1)
            # attr_norm = np.squeeze(MinMaxScaler().fit_transform(attr_norm))
            attr_norm = np.linalg.norm(attr, axis=1)
            # attr_norm = softmax(attr_norm)
            temporary_list.append(np.mean(attr_norm * np.array(emb).T, axis=1))
            norm_list.extend(list(attr_norm))
        return np.array(temporary_list)

    @staticmethod
    def emb_dot_attr(attr_obj, emb_obj):
        temporary_list = []
        for attr, emb in zip(attr_obj, emb_obj):
            attr, emb = np.array(attr), np.array(emb)
            temporary_list.append(np.mean(attr * emb, axis=0))
        return np.array(temporary_list)

    # カウントベクトルの重み付け
    @staticmethod
    def counter(attr_obj, token_obj):
        dict_list = []
        char_list = []
        for attr, token, in zip(attr_obj, token_obj):
            part_dict = defaultdict(int)
            assert len(token) == len(attr)
            for idx in range(len(token)):
                part_dict[token[idx]] += attr[idx]
            dict_list.append(part_dict)
            char_list.extend(token)

        unique_character_list = list(set(char_list))
        vectorize_dict = {char: idx for idx, char in enumerate(unique_character_list)}
        temporary_list = []
        for attr, token, in zip(attr_obj, token_obj):
            vector = np.zeros(len(unique_character_list))
            for token_idx in range(len(token)):
                vector_idx = vectorize_dict[token[token_idx]]
                vector[vector_idx] += attr[token_idx]
            temporary_list.append(vector / np.linalg.norm(vector))

        result_tensor = np.array(temporary_list)
        # scaling
        scaler = Normalizer()
        scaled_data = scaler.fit_transform(result_tensor)
        a = np.linalg.norm(scaled_data, axis=1)

        return scaled_data
