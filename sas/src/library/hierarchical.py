import sys
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import *
import collections
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("..")
from library.selector import Selector

class HierarchicalClustering:
    def __init__(self):
        self.hierarchy = None

    def clustering(self, attributions, k=10):
        attr = attributions
        hierarchy = linkage(attr, method="ward")
        cluster = fcluster(hierarchy, t=100, criterion="distance")
        return hierarchy, cluster, attr

    def linkage(self, attributions):
        self.hierarchy = linkage(attributions, method="ward")
        return self.hierarchy

    def fcluster(self, t, criterion="maxclust"):
        cluster = fcluster(self.hierarchy, t=t, criterion=criterion)
        return cluster

    def dendrogram(self, dir_path, script_name, term):
        fig, ax = plt.subplots(1, 1,)
        dn = dendrogram(self.hierarchy, p=7, truncate_mode='level', ax=ax)
        plt.tight_layout()
        plt.savefig(Path(dir_path) / "{}_{}_dendro.png".format(script_name, term))

    def to_tree(self):
        root_node = to_tree(self.hierarchy, rd=False)

        def func(node):
            if node.is_leaf():
                return node.pre_order()
            else:
                return [node.pre_order(), [func(node.get_left()), func(node.get_right())]]

        foo = func(root_node)

        return to_tree(self.hierarchy, rd=False)


def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el