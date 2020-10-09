# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../')

from cnm import Clustering, TransitionProperties, Propagation


def runLorenz():

    # Create the Lorenz data
    from helper import create_lorenz_data
    #data, dt = create_lorenz_data()
    import numpy as np
    data = np.load('/home/daniel/Documents/Thesis/CNM/runDS/Lorenz/SOC-WithPast/francki/Data/Rho28/Data.npy')
    print(data.shape)

    #exit()

    # CNM parameters:
    K = 50 # Number of clusters
    L = 23 # Model order
    K = 5
    L = 2

    # Clustering
    from sklearn.cluster import KMeans
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
            }
    
    clustering = Clustering(**cluster_config)
    
    # Transition properties
    transition_config = {
            'clustering': clustering,
            'dt': dt,
            'K': K,
            'L': L,
            }
    
    transition_properties = TransitionProperties(**transition_config)
    
    
    #815         KM = KMeans(
    #816                 n_clusters=self.nClusters,
    #817                 max_iter=1000,
    #818                 n_init=100,
    #819                 n_jobs=-1,
    #820                 )
    #821
    #822         # --> Fit the KMeans
    #823         foo = ti.time()
    #824         KM.fit(Ampl)
    #825
    #826         # --> Get the relevant attributes
    #827         self.KMeans = KM
    #828         return KM.labels_, KM.cluster_centers_
    
    
    
    ## Perform clustering
    #cluster_options = {
    #    Data foo,
    #    opt2: bar,
    #    algo=kmeans(K,...)
    #    }
    #
    #cluster = Cluster(**cluster_options)
    #process = MarkovProcess(**process_options,cluster)
    #cnm = CNM(**...,process)
    #cnm.run(tTotal,IC)

if __name__== '__main__':
    runLorenz()
