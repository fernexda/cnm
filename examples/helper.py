# -*- coding: utf-8 -*-

import numpy as np

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
    T = 100                     # total time
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
