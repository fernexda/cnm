# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Daniel Fernex.
# Copyright (c) 2020 Bernd R. Noack.
# Copyright (c) 2020 Richard Semaan.
#
# This file is part of CNM 
# (see https://github.com/fernexda/cnm).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import sys
sys.path.insert(0,'../')
from cnm import Clustering, TransitionProperties, Propagation
import numpy as np

def run_ecg():

    from helper import create_lorenz_data
    from sklearn.cluster import KMeans

    # CNM parameters:
    # ---------------
    K = 50 # Number of clusters
    L = 23 # Model order

    # Create the Lorenz data
    case_data = np.load('data/ecg.npz')
    data, dt = case_data['data'], case_data['dt']
    t = np.arange(data.shape[0]) * dt

    # Scale inputs for a better clustering (amplitude of data[:,1] is two orders
    # of magnitude larger than data[:,0])
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(data)

    import matplotlib.pyplot as plt

    # Clustering
    # ----------
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=1000,n_init=100,n_jobs=-1),
            'dataset': 'ecg', # To store the centroids properly
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
    t_total = 16.5
    dt_hat = dt # To spline-interpolate the centroid-to-centroid trajectory

    propagation = Propagation(**propagation_config)
    t_hat, x_hat = propagation.run(t_total,ic,dt_hat)

    # Plot the results
    # ----------------
    from helper import (plot_phase_space, plot_time_series,plot_cpd,
                        plot_autocorrelation)

    ## phase space
    #dim = '2d'
    #plot_phase_space(data,clustering.centroids,clustering.labels,dim=dim)

    # time series
    time_range = (0,10)
    n_dim = 1
    plot_label = ['V']
    plot_time_series(t,data,t_hat,x_hat,time_range,plot_label,n_dim)

    # cluster probability distribution
    plot_cpd(data,x_hat)

    # autocorrelation function
    time_blocks = 40
    time_range = (-0.5,14)
    plot_autocorrelation(t,data,t_hat,x_hat,time_blocks,time_range)

if __name__== '__main__':
    run_ecg()
