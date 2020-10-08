# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../cnm')
print(sys.path)

#from cnm import Cluster#, MarkovProcess, CNM
#import Cluster
from cluster import Cluster

# Run the Lorenz system

# Create the Lorenz data
from helper import create_lorenz_data
data = create_lorenz_data()

# CNM parameters:
K = 50 # Number of clusters
L = 23 # Model order

# Clustering
from sklearn.cluster import KMeans
cluster_config = {
        'data': data,
        'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
        }

cluster = Cluster(cluster_config)

# Transition properties
process_config = {
        'cluster': cluster,

process = MarkovProcess(


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
