# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../')
from cnm import Clustering, TransitionProperties, Propagation
import numpy as np
from helper import create_roessler_data
from sklearn.cluster import KMeans

def run_roessler():


    # CNM parameters:
    # ---------------
    K = 100 # Number of clusters
    L = 1 # Model order

    # Create the Lorenz data
    data, dt = create_roessler_data()
    t = np.arange(data.shape[0]) * dt

    # Clustering
    # ----------
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
            'dataset': 'roessler',
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
    t_total = 500
    dt_hat = dt # To spline-interpolate the centroid-to-centroid trajectory

    propagation = Propagation(**propagation_config)
    t_hat, x_hat = propagation.run(t_total,ic,dt_hat)

    # Plot the results
    # ----------------
    from helper import (plot_phase_space, plot_time_series,plot_cpd,
                        plot_autocorrelation)

    # phase space
    plot_phase_space(data,clustering.centroids,clustering.labels)

    # time series
    time_range = (0,100)
    plot_label = ['x','y','z']
    plot_time_series(t,data,t_hat,x_hat,time_range)

    # cluster probability distribution
    plot_cpd(data,x_hat)

    # autocorrelation function
    time_blocks = 100
    time_range = [0,80]
    method = 'dot'
    plot_autocorrelation(t,data,t_hat,x_hat,time_blocks,time_range,method=method)

if __name__== '__main__':
    run_roessler()
