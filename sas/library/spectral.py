import numpy as np
import sklearn.neighbors
from tqdm import tqdm
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse.csgraph import laplacian
from scipy.cluster.hierarchy import *
from sklearn.cluster import AgglomerativeClustering


class SpectralClustering:
    def cosine_similarity(self, i_vector, j_vector):
        similarity = cosine_similarity([i_vector], [j_vector])[0][0]
        return similarity

    def affinity_from_embeddings(self, attributions):
        matrix = np.zeros((len(attributions), len(attributions)), dtype=np.float)
        summation_list = [np.sum(attr, axis=0) for attr in tqdm(attributions)]
        for i_idx, i_attr in enumerate(tqdm(attributions)):
            for j_idx, j_attr in enumerate(attributions):
                similarity = self.cosine_similarity(summation_list[i_idx], summation_list[j_idx])
                matrix[i_idx, j_idx] = similarity

        np.fill_diagonal(matrix, 0.0)
        matrix = np.nan_to_num(matrix)
        return matrix

    def nearest_neighbor_graph(self, X, affine=False, n_neighbors=10):
        '''
        Calculates nearest neighbor adjacency graph.
        https://en.wikipedia.org/wiki/Nearest_neighbor_graph
        '''
        A = X

        # sort each row by the distance and obtain the sorted indexes
        sorted_rows_ix_by_dist = np.argsort(A, axis=1)

        # pick up first n_neighbors for each point (i.e. each row)
        # start from sorted_rows_ix_by_dist[:,1] because because sorted_rows_ix_by_dist[:,0] is the point itself
        # for smaller datasets use sqrt(#samples) as n_neighbors. max n_neighbors = 10
        n_neighbors = min(int(np.sqrt(A.shape[0])), n_neighbors)
        nearest_neighbor_index = sorted_rows_ix_by_dist[:, 1:n_neighbors+1]

        # initialize an nxn zero matrix
        W = np.zeros(A.shape)

        # for each row, set the entries corresponding to n_neighbors to 1
        for row in range(W.shape[0]):
            W[row, nearest_neighbor_index[row]] = 1

        # make matrix symmetric by setting edge between two points
        # if at least one point is in n nearest neighbors of the other
        for r in range(W.shape[0]):
            for c in range(W.shape[0]):
                if W[r, c] == 1:
                    W[c, r] = 1

        return W

    def compute_laplacian(self, W, d=None):
        # calculate row sums
        if d is None:
            d = W.sum(axis=1)

        D = np.diag(d)

        #create degree matrix
        L = D - W
        D_inv = np.sqrt(np.linalg.inv(D))
        L_sym = np.dot(D_inv, np.dot(L, D_inv))

        return L_sym

    def get_eigvecs(self, L):
        eigvals, eigvecs = np.linalg.eig(L)
        eigvals, eigvecs = eigvals.real, eigvecs.real
        return eigvals, eigvecs

    def select_eigvecs(self, eigvals, eigvecs, k):
        # sort eigenvalues and select k smallest values - get their indices
        ix_sorted_eig = np.argsort(eigvals)[:k]

        # calculate eigan gap
        sorted_eig = np.sort(eigvals)
        eigen_gap = [sorted_eig[i] - sorted_eig[i-1] for i in range(1, len(sorted_eig))]

        # select k eigenvectors corresponding to k-smallest eigenvalues
        return sorted_eig, eigvecs[:, ix_sorted_eig]

    def spectral_clustering(self, X, k):
        A = self.affinity_from_embeddings(attributions=X)
        W = self.nearest_neighbor_graph(A, n_neighbors=5)
        L = self.compute_laplacian(W)
        eig_vals, eig_vecs = self.get_eigvecs(L)
        k_eig_vals, k_eig_vecs = self.select_eigvecs(eig_vals, eig_vecs, k)
        k_means = KMeans(n_clusters=k, random_state=0).fit(k_eig_vecs)
        cluster = k_means.labels_.tolist()
        # hierarchy
        centers = k_means.cluster_centers_
        hierarchy = linkage(centers, "ward")
        return cluster, k_eig_vals, k_eig_vecs, hierarchy

    def hierarchical_spectral_clustering(self, X, k):
        A = self.affinity_from_embeddings(attributions=X)
        W = self.nearest_neighbor_graph(A, n_neighbors=5)
        L = self.compute_laplacian(W)
        eig_vals, eig_vecs = self.get_eigvecs(L)
        k_eig_vals, k_eig_vecs = self.select_eigvecs(eig_vals, eig_vecs, k)
        hierarchy = linkage(k_eig_vecs, "ward")
        return hierarchy, k_eig_vals, k_eig_vecs

    def spectral_clustering_generator(self, X, affine=True, k_range=(2, 10, 1)):
        # create weighted adjacency matrix
        A = self.affinity_from_embeddings(attributions=X)
        W = self.nearest_neighbor_graph(A, affine, n_neighbors=5)

        # W = sklearn.neighbors.kneighbors_graph(X, 10).toarray()
        L = self.compute_laplacian(W)

        # create projection matrix with first k eigenvectors of L
        eig_vals, eig_vecs = self.get_eigvecs(L)

        for k in tqdm(range(*k_range)):
            k_eig_vals, k_eig_vecs = self.select_eigvecs(eig_vals, eig_vecs, k)
            k_means = KMeans(n_clusters=k, random_state=0).fit(k_eig_vecs)
            cluster = k_means.labels_.tolist()
            yield k, cluster, k_eig_vals, k_eig_vecs


# deprecated
class PreprocessForClustering:
    @staticmethod
    def padding_1d(attributions):
        max_length = max([len(a) for a in tqdm(attributions)])
        padded = []
        for attr in attributions:
            attr = np.array(attr)
            pad = np.zeros((max_length - attr.shape[0],))
            padded.append(np.concatenate([attr, pad]))
        return np.array(padded)

    @staticmethod
    def padding_2d(attributions):
        max_length = max([len(a) for a in tqdm(attributions)])
        padded = []
        for attr in attributions:
            attr = np.array(attr)
            pad = np.zeros((max_length - attr.shape[0], attr.shape[1]))
            padded.append(np.concatenate([attr, pad]))
        return np.array(padded)

    @staticmethod
    def matrix_based_counter(tokens, attributions, counter=False):
        unique_char_list = np.unique(sum(tokens, [])).tolist()
        char_to_idx = {char: idx for idx, char in enumerate(unique_char_list)}
        vec_size = len(char_to_idx)
        matrix = np.zeros((len(tokens), vec_size))
        for col, (token, attribution) in enumerate(zip(tokens, attributions)):
            for idx in range(len(token)):
                char, attr = token[idx], attribution[idx]
                matrix[col, char_to_idx[char]] += attr if not counter else 1.0
        return matrix

    @staticmethod
    def affine_matrix_based_top_features(tokens, attributions, top_k=5):
        matrix = np.zeros((len(attributions), len(attributions)))
        for idx, (i_token, i_attr) in tqdm(enumerate(zip(tokens, attributions))):
            i_top_arg = np.argsort(i_attr)[::-1][:top_k]
            i_top_token = np.array(i_token)[i_top_arg]
            for jdx, (j_token, j_attr) in enumerate(zip(tokens, attributions)):
                j_top_arg = np.argsort(j_attr)[::-1][:top_k]
                j_top_token = np.array(j_token)[j_top_arg]
                agree_rate = len(set(i_top_token) & set(j_top_token)) / top_k
                matrix[idx, jdx] = 1.0 - agree_rate
        return matrix

    @staticmethod
    def _generate_count_dict(token, attr):
        i_dict = defaultdict(float)
        for i_t, i_a in zip(token, attr):
            i_dict[i_t] += i_a
        return i_dict

    @staticmethod
    def _generate_bow_vector(i_dict, j_dict):
        included_token = list(set(i_dict.keys()) | set(j_dict.keys()))
        vector_length = len(included_token)
        i_vector, j_vector = np.zeros(vector_length), np.zeros(vector_length)
        for idx, token in enumerate(included_token):
            i_vector[idx] = i_dict[token]
            j_vector[idx] = j_dict[token]
        return i_vector, j_vector

    @staticmethod
    def _to_bow_vector(dict_list):
        char_set = set()
        for _dict in dict_list:
            char_set |= set(_dict.keys())
        char_list = list(char_set)
        char_to_idx = {c: i for i, c in enumerate(char_list)}

        vector_list = []
        for i_dict in dict_list:
            vector = np.zeros(len(char_list))
            for char, attr in i_dict.items():
                idx = char_to_idx[char]
                vector[idx] += attr
            vector_list.append(vector)

        return vector_list

    @staticmethod
    def _cosine_similarity(i_vector, j_vector):
        similarity = cosine_similarity([i_vector], [j_vector])[0][0]
        return similarity
        # return np.dot(i_vector, j_vector) / (np.linalg.norm(i_vector) * np.linalg.norm(j_vector))

    @staticmethod
    def affine_based_bow(tokens, attributions):
        matrix = np.zeros((len(tokens), len(tokens)), dtype=np.float)
        dict_list = [PreprocessForClustering._generate_count_dict(token, attr) for token, attr in zip(tokens, attributions)]
        vector_list = PreprocessForClustering._to_bow_vector(dict_list)
        for i_idx, i_vector in enumerate(vector_list):
            for j_idx, j_vector in enumerate(vector_list):
                similarity = PreprocessForClustering._cosine_similarity(i_vector, j_vector)
                matrix[i_idx, j_idx] = similarity

        return matrix

    @staticmethod
    def affine_embedding(tokens, attributions):
        matrix = np.zeros((len(tokens), len(tokens)), dtype=np.float)
        summation_list = [np.sum(attr, axis=0) for attr in tqdm(attributions)]
        for i_idx, i_attr in enumerate(tqdm(attributions)):
            for j_idx, j_attr in enumerate(attributions):
                similarity = PreprocessForClustering._cosine_similarity(summation_list[i_idx], summation_list[j_idx])
                matrix[i_idx, j_idx] = similarity

        np.fill_diagonal(matrix, 0.0)
        matrix = np.nan_to_num(matrix)
        return matrix

    @staticmethod
    def embedding_simple(tokens, attributions):
        matrix = PreprocessForClustering.padding_1d(attributions)
        return matrix

