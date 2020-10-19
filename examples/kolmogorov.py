# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../')
from cnm import Clustering, TransitionProperties, Propagation
import numpy as np
from helper import create_roessler_data
from sklearn.cluster import KMeans

def run_kolmogorov():


    # CNM parameters:
    # ---------------
    K = 200 # Number of clusters
    L = 24 # Model order

    # Create the Lorenz data
    case_data = np.load('data/kolmogorov.npz')
    data, dt = case_data['data'], case_data['dt']
    t = np.arange(data.shape[0]) * dt

    # Clustering
    # ----------
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
            'dataset': 'kolmogorov',
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
    t_total = 10000
    dt_hat = dt # To spline-interpolate the centroid-to-centroid trajectory

    propagation = Propagation(**propagation_config)
    t_hat, x_hat = propagation.run(t_total,ic,dt_hat)

    # Plot the results
    # ----------------
    from helper import (plot_phase_space, plot_time_series,plot_cpd,
                        plot_autocorrelation)

    # phase space
    n_dim = 2
    plot_phase_space(data,clustering.centroids,clustering.labels,n_dim=n_dim)

    # time series
    time_range = (0,3000)
    n_dim = 1
    plot_label = ['D']
    plot_time_series(t,data,t_hat,x_hat,time_range,plot_label,n_dim=n_dim)

    # cluster probability distribution
    plot_cpd(data,x_hat)

    # autocorrelation function
    time_blocks = t_hat[-1]
    time_range = [0,200]
    plot_autocorrelation(t,data,t_hat,x_hat,time_blocks,time_range)

if __name__== '__main__':
    run_kolmogorov()
