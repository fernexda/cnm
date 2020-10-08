# -*- coding: utf-8 -*-

class Clustering:
    """
    """

    def __init__(self,data,cluster_algo):

        #cluster_algo = cluster_config['cluster_algo']
        #data = cluster_config['data']

        cluster_algo.fit(data)

        self.labels = cluster_algo.labels_
        self.centroids = cluster_algo.cluster_centers_
