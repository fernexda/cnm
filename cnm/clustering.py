# -*- coding: utf-8 -*-

import numpy as np

class Clustering:
    """Perform the data clustering with the requested clustering algorithm.

    Parameters
    ----------

    data: ndarray of shape (n_snapshots,n_dim)
        Snapshots of the dynamical system, equally spaced in time.

    cluster_algo: object
        Instance from the chosen clustering class. Must contain the 'fit()'
        method and return labels_ and cluster_centers_.

    Attributes
    ----------

    labels: ndarray of shape (n_snapshots,)
        Cluster affiliation of each snapshot.

    centroids: ndarray of shape (K,n_dim)
        Centroids of the clusters.

    cluster_sequence: ndarray of shape (# transition+1,)
        Sequence of visited clusters.
    """

    def __init__(self,data,cluster_algo):

        # Perform clustering
        cluster_algo.fit(data)

        self.labels = cluster_algo.labels_
        self.centroids = cluster_algo.cluster_centers_
        diff = np.diff(self.labels)
        self.cluster_sequence = self.labels[np.insert(diff.astype(np.bool), 0, True)]

if __name__=='__main__':

    from sklearn.cluster import KMeans
    import numpy as np
    np.random.seed(0)

    # Number of clusters
    K = 5

    # get test data
    data = np.load('test_data/data.npy')
    centroids_test = np.loadtxt('test_data/centroids-K{}'.format(K))
    labels_test = np.loadtxt('test_data/labels-K5'.format(K))

    # Perform clustering
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=1000,n_init=100,n_jobs=-1),
            }
    clustering = Clustering(**cluster_config)

    # Check clustering
    assert np.all(clustering.centroids == centroids_test)

    # Check labels
    assert np.all(clustering.labels == labels_test)
