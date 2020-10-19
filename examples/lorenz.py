# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../')
from cnm import Clustering, TransitionProperties, Propagation
import numpy as np

def run_lorenz():

    from helper import create_lorenz_data
    from sklearn.cluster import KMeans

    # CNM parameters:
    # ---------------
    K = 50 # Number of clusters
    L = 22 # Model order

    # Create the Lorenz data
    data, dt = create_lorenz_data()
    t = np.arange(data.shape[0]) * dt

    # Clustering
    # ----------
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
            'dataset': 'lorenz',
            }

    clustering = Clustering(**cluster_config)

    # Transition properties
    # ---------------------
    transition_config = {
            'clustering': clustering,
            'dt': dt,
            'K': K,
            'L': L,
            }

    transition_properties = TransitionProperties(**transition_config)

    # Propagation
    # -----------
    propagation_config = {
            'transition_properties': transition_properties,
            }

    ic = 0 # Index of the centroid to start in
    t_total = 950
    dt_hat = dt # To spline-interpolate the centroid-to-centroid trajectory

    propagation = Propagation(**propagation_config)
    t_hat, x_hat = propagation.run(t_total,ic,dt_hat)

    # Plot the results
    # ----------------
    from helper import (plot_phase_space, plot_time_series,plot_cpd,
                        plot_autocorrelation)

    # phase space
    plot_phase_space(data,clustering.centroids,clustering.labels)#,n_dim)

    # time series
    time_range = (45,60)
    n_dim = 3
    plot_time_series(t,data,t_hat,x_hat,time_range,n_dim)

    # cluster probability distribution
    plot_cpd(data,x_hat)

    # autocorrelation function
    time_blocks = 40
    time_range = (-0.5,time_blocks)
    plot_autocorrelation(t,data,t_hat,x_hat,time_blocks,time_range)

if __name__== '__main__':
    run_lorenz()
