# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'../')
from cnm import Clustering, TransitionProperties, Propagation
import numpy as np

def runLorenz():

    from helper import create_lorenz_data
    from sklearn.cluster import KMeans

    # CNM parameters:
    # ---------------
    K = 50 # Number of clusters
    L = 23 # Model order

    # Create the Lorenz data
    import os
    import pickle
    if not os.path.exists('output/lorenz/cnm_data.npz'):
        data, dt = create_lorenz_data()
        t = np.arange(data.shape[0]) * dt

        # Clustering
        # ----------
        cluster_config = {
                'data': data,
                'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=10,n_jobs=-1),
                }

        clustering = Clustering(**cluster_config)

        with open('output/lorenz/clustering.pickle','wb') as f:
            pickle.dump(clustering,f)


        # SAVE
        np.savez(
                'output/lorenz/cnm_data',
                data = data,
                t = t,
                dt = dt,
                )
    else:
        with open('output/lorenz/clustering.pickle','rb') as f:
            clustering = pickle.load(f)
        D = np.load('output/lorenz/cnm_data.npz')
        data = D['data']
        t = D['t']
        dt = D['dt']

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

    ## phase space
    #plot_phase_space(data,clustering.centroids,clustering.labels)

    ## time series
    #plot_time_series(t,data,t_hat,x_hat)

    # cluster probability distribution
    #plot_cpd(data,x_hat)

    # autocorrelation function
    plot_autocorrelation(t,data,t_hat,x_hat)

if __name__== '__main__':
    runLorenz()
