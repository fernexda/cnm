# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm

FIGSIZE = (8,7)

def plot_phase_space(data,centroids,labels):

    n_cl = centroids.shape[0]

    plt.close()
    fig = plt.figure(figsize=(FIGSIZE))
    ax = fig.add_subplot(111,projection='3d')

    # Data line (grey). For clarity, show only part of the trajectory.
    n_snap = 3000
    ax.plot(
            data[:2*n_snap,0],
            data[:2*n_snap,1],
            data[:2*n_snap,2],
            '-',
            alpha=0.5,
            c='grey',
            zorder=0,
            linewidth=0.5,
            )

    # snapshots with their affiliation. For clarity, show only part of the snapshots
    colors = cm.jet(np.linspace(0,1,n_cl))
    ax.scatter(
            data[::2][:n_snap,0],
            data[::2][:n_snap,1],
            data[::2][:n_snap,2],
            c=colors[labels[::2][:n_snap]],
            label='Data',
            zorder=1,
            alpha=0.5,
            s=5,
            )

    # Centroids
    ax.plot(
            centroids[:,0],
            centroids[:,1],
            centroids[:,2],
            'o',
            color='k',
            zorder=5,
            markersize=7.5,
            )

    # Background and no axes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()

    # Hide grid lines
    ax.grid(False)

    # show the plot
    plt.show()


def plot_time_series():
    pass
def plot_cpd():
    pass
def plot_autocorrelation():
    pass

def create_lorenz_data():
    """Create the Lorenz data"""

    from scipy.integrate import solve_ivp

    # Lorenz settings
    sigma = 10
    rho   = 28
    beta  = 8/3.
    x0,y0,z0 = (-3,0,31) # Initial conditions
    np.random.seed(0)

    # Lorenz system
    def lorenz(t,q,sigma,rho,beta):
        return [
                sigma * (q[1] - q[0]),
                q[0] * (rho - q[2]) - q[1],
                q[0] * q[1] - beta * q[2],
                ]

    # Time settings
    T = 1000                     # total time
    n_points = T * 60            # number of samples
    #T = 1000                     # total time
    #n_points = T * 60            # number of samples
    t = np.linspace(0,T,n_points)# Time vector
    dt = t[1]-t[0]               # Time step

    # integrate the Lorenz system
    solution = solve_ivp(fun=lambda t, y: lorenz(t, y, sigma,rho,beta), t_span = [0,T], y0 = [x0,y0,z0],t_eval=t)
    data = solution.y.T

    # remove the first 5% to keep only the 'converged' part which is in the ears
    points_to_remove = int(0.05 * n_points)
    return data[points_to_remove:,:], dt



