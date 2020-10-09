# -*- coding: utf-8 -*-

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
    """

    def __init__(self,data,cluster_algo):

        # Perform clustering
        cluster_algo.fit(data)

        self.labels = cluster_algo.labels_
        self.centroids = cluster_algo.cluster_centers_

if __name__=='__main__':

    from sklearn.cluster import KMeans
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
            }

    clustering = Clustering(**cluster_config)

    cl = Clustering
