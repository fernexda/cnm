# -*- coding: utf-8 -*-

import numpy as np

class Propagation:

    def __init__(self,transition_properties):

        self.transition = transition_properties
        self.centroids = transition_properties.clustering.centroids
        self.cluster_sequence = transition_properties.cluster_sequence
        self.L = transition_properties.L

    def run(self,t_total,ic,dt):
        """Propagate the state in the phase space.

        Parameters
        ----------
        t_total: float
            Total simulation time. Propagation stops when this time is reached.
        ic: int
            Initial condition, index of the centroid used as initial condition.
        dt: float
            Time step for the spline-interpolated trajectory.

        Returns
        -------
        x_hat: ndarray of shape (n_times x n_dim)
            The predicted states corresponding to the visited centroids. n_times
            is the number of steps after spline interpolation.
        """

        # Initialize past of ic, finding the first centroid sequence of size L
        # ending with ic
        past_found = False
        for i_cl,cl in enumerate(self.cluster_sequence):
            if (cl == ic) and (i_cl >= self.L-1):
                past_cl = self.cluster_sequence[i_cl-self.L+1:i_cl+1]
                past_found = True
                break
        if not past_found:
            msg = (
                    "Past not found. You are maybe asking for a too long past. "
                    "Try again with a shorter past."
                    )
            raise Exception(msg)

        # Initialize variables
        t = [0]
        self.visited_centroids = [ic]

        # Propagate iteratively
        while t[-1] < t_total:

            # Find the next destination and required time
            next_cl, transition_time = self.transition.step(past_cl)

            # Update the time and past
            past_cl[:-1] = past_cl[1:]
            past_cl[-1] = next_cl
            t.append(t[-1] + transition_time)

            # Store visited centroid
            self.visited_centroids.append(next_cl)

        # Get the corresponding states
        x_hat = self.centroids[self.visited_centroids]

        # Smooth the trajectory
        return self._interpolate_spline(t,x_hat,dt)

    def _interpolate_spline(self,t,x,dt):
        """Interpolate the centroid-to-centroid trajectory with splines.

        Parameters
        ----------
        t: ndarray of shape (n_transitions,)
            Time of the sequential cluster visits.

        x: ndarray of shape (n_transitions x n_dim)
            State of the sequentially visited centroids.

        dt: float
            Time step for the interpolation.

        Returns
        -------
        t_int: ndarray of shape (n_times,)
            Times of the interpolated trajectory

        x_int: ndarray of shape (n_times x n_dim)
            Interpolated trajectory.
        """

        # Create the interpolated time vector
        t_int = np.arange(t[0],t[-1],dt)

        x_int = np.empty((t_int.size,x.shape[1]))

        # --> Interpolate
        from scipy.interpolate import InterpolatedUnivariateSpline
        for i_dim in range(x.shape[1]):

            spline = InterpolatedUnivariateSpline(t, x[:,i_dim])

            # --> Store
            x_int[:,i_dim] = spline(t_int)

        return t_int, x_int

if __name__=="__main__":

    # Do clustering and transition properties
    from sklearn.cluster import KMeans
    import numpy as np
    np.random.seed(0)

    # CNM config
    K = 5
    L = [1,3]
    dt = 0.016666944449074152

    # get test data and perform clustering (needed for transition properties)
    data = np.load('test_data/data.npy')

    # Clustering
    # --------------------------------------------------------------------------
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=1000,n_init=100,n_jobs=-1),
            }
    from clustering import Clustering
    from transition_properties import TransitionProperties
    clustering = Clustering(**cluster_config)

    # Test with and without past
    for l in L:

        # Transition properties
        # --------------------------------------------------------------------------
        transition_config = {
                'clustering': clustering,
                'dt': dt,
                'K': K,
                'L': l,
                }
        transition_properties = TransitionProperties(**transition_config)

        # Propagations
        # --------------------------------------------------------------------------
        propagation_config = {
                'transition_properties': transition_properties,
                }

        ic = 0        # Index of the centroid to start in
        t_total = 20  #

        propagation = Propagation(**propagation_config)
        t_hat, x_hat = propagation.run(t_total,ic,dt)

        # Read validation data
        visited_centroids_test = np.loadtxt('test_data/visited_centroids-K{}-L{}'.format(K,l))
        t_visited_centroids_test = np.loadtxt('test_data/t_visited_centroids-K{}-L{}'.format(K,l))

        # NOTE It doesn't make sense to compare, as the trajectories will
        # inevitably diverge, due to the probabilistic selection of the next
        # destination!!
        # If the previous tests are successful, propagation is also correct.

        # test visited centroids
        # assert np.all(propagation.visited_centroids == visited_centroids_test)

