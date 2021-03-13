import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import NearestNeighbors

from src.utils.common import timeit


@timeit
def plot_dendrogram(model, **kwargs):
    """ Create linkage matrix and then plot the dendrogram
        source:
        https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


@timeit
def plot_sorted_nn_dists(df, min_pts):
    plt.figure()
    neighbors_fit = NearestNeighbors(n_neighbors=min_pts).fit(df)
    distances, indices = neighbors_fit.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
