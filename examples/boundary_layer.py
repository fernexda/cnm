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
from helper import create_roessler_data
from sklearn.cluster import KMeans

def run_boundary_layer():

    # CNM parameters:
    # ---------------
    K = 50 # Number of clusters
    L = 3 # Model order

    # Create the Lorenz data
    case_data = np.load('data/boundary_layer_l100_t120_a60.npz')
    data, dt = case_data['data'], case_data['dt']
    t = np.arange(data.shape[0]) * dt

    # Clustering
    # ----------
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=300,n_init=100,n_jobs=-1),
            'dataset': 'boundary_layer',
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

    ic = 36       # Index of the centroid to start in
    t_total = 840 # Total simulation time
    dt_hat = dt   # To spline-interpolate the centroid-to-centroid trajectory

    propagation = Propagation(**propagation_config)
    t_hat, x_hat = propagation.run(t_total,ic,dt_hat)

    # Plot the results
    # ----------------
    from helper import (plot_phase_space, plot_time_series,plot_cpd,
                        plot_autocorrelation)

    # phase space
    n_dim = 3
    plot_phase_space(data,clustering.centroids,clustering.labels,n_dim=n_dim)

    # time series
    time_range = (0,t[-1])
    n_dim = 3
    plot_label = ['a_1','a_2','a_3']
    plot_time_series(t,data,t_hat,x_hat,time_range,plot_label,n_dim=n_dim)

    # cluster probability distribution
    plot_cpd(data,x_hat)

    # autocorrelation function
    time_blocks = t_hat[-1]
    time_range = [0,390]
    method = 'dot'
    plot_autocorrelation(t,data,t_hat,x_hat,time_blocks,time_range,method=method)

if __name__== '__main__':
    run_boundary_layer()
