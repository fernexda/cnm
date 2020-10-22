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

# -*- coding: utf-8 -*-

import numpy as np
from itertools import groupby
from sklearn.neighbors import KDTree


class TransitionProperties:
    """Compute the direct transition probability Q and the transition time T

    Attributes
    ----------
    clustering : instance
        Instance from the Clustering class.
    K : int
        Number of clusters.
    L : int
        CNM model order
    dt : float
        Time step of the data.
    labels : ndarray of shape (n_snapshots,)
        Cluster affiliation of each snapshot.
    centroids : ndarray of shape (K,n_dim)
        Centroids of the clusters.
    cluster_sequence : ndarray of shape (# transition+1,)
        Sequence of visited clusters.
    Q : dict
        Transition probabilities for an L-order model.  The keys of Q are string
        of the past centroids. If the previously visited centroids are 3
        (newest), 2, and 1 (oldest), the key will be '1,2,3'. The corresponding
        values are 2D arrays, where the first column is the index of the
        possible destination centroid and the 2 column is the corresponding
        probability.
    T : dict
        Transition times for an L-order model. The keys of T are string
        of the past centroids and the future one. If the previously visited centroids are 3
        (newest), 2, and 1 (oldest), and the next destination is 4, the key will
        be '1,2,3,4'. The corresponding
        values are the transition time of the transition 3->4, having visited 1
        and 2 before..

    Notes
    -----
    If the labels are [0,1,1,2,2,2,0,3,3,4,4,4], the model order is L=3:
    Q = {
            '0,1,2': [0,1],
            '1,2,0': [3,1],
        }
    T = {
            '0,1,2,0': Tau_(2->0),
            '1,2,0,3': Tau_(0->3),
        }
    The requests to T are always done with one more key than the requests to Q.
    The transition to the final cluster is neglected, because the transition is
    not complete, so the corresponding time would be wrong.
    """

    def __init__(self, clustering, K: int, L: int, dt):

        print('Identify the transition properties')
        print('----------------------------------')
        print('Model order: {}'.format(L))

        self.clustering = clustering
        self.centroids = clustering.centroids
        self.labels = clustering.labels
        self.cluster_sequence = clustering.cluster_sequence
        self.K = K
        self.L = L
        self.dt = dt

        # Safety check
        if self.L <= 0:
            raise Exception('The model order must be > 0')

        print('Compute Q')
        self.Q = self._compute_Q()

        print('Compute T')
        self.T = self._compute_T()

        print('\n')

    def step(self,past_cl):
        """Find the next centroid and corresponding transition time.

        Parameters
        ----------
        past_cl : list of length L
            Contains the current and previous centroids. With L=3,
            past_cl=[c_i,c_j,c_k] (in the case where c_l is the destination)

        Returns
        -------
        past_cl : list of length L
            Contains the current and previous centroids. Note that they might
            differ from the input parameter `past_cl` in the case where
            past_cl[-1] has not possible destination. In that case, the method 
            _get_next_cl_from_neighbor(past_cl) will find a new `past_cl`, as
            close as possible to the original one.
        next_cl : int
            Index of the next cluster
        transition_time : float
            Transition time to transit from past_cl[-1] to next_cl.
        """

        # Select next cluster
        try:
            key = ','.join(map(str, past_cl))
            next_cl = int(np.random.choice(self.Q[key][:,0], p=self.Q[key][:,1]))

        except KeyError:

            # The current centroid has no next centroid (data is too short or
            # too many centroids)
            past_cl = self._get_next_cl_from_neighbor(past_cl)
            key = ','.join(map(str, past_cl))

            next_cl = int(np.random.choice(self.Q[key][:,0], p=self.Q[key][:,1]))

        # Read the corresponding transition time
        key = ','.join(map(str, past_cl))+',{}'.format(str(next_cl))
        transition_time = self.T[key]

        return past_cl, next_cl, transition_time

    def _get_next_cl_from_neighbor(self,past_cl):
        """Finds the next destination centroid from another trajectory.

        In case where the data time range is too short or there are too many
        centroids, it can be that the centroid past_cl[-1] has no destination. This means the
        data simply stops in this centroid and doesn't go further, i.e., there cannot
        by any transition from there. In this case, an alternative
        destination past_cl is found, consistent with the current trajectory,
        from which a next_cl can be determined.

        The steps are:

        1) Find nearest centroid to the current centroid (past_cl[-1])
        2) Find a past of this centroids as similar as possible to the current
           trajectory
        3) Use this past to replace the current past
        This causes typically a small glitch in the trajectory but allows the
        system to propagate indefinitely.

        Attributes
        ----------
        past_cl : list of length L
            Contains the current and previous centroids. With L=3,
            past_cl=[c_i,c_j,c_k] (in the case where c_l is the destination).
            Note that past_cl[-1] has no possible destination.

        Returns
        -------
        possible_pasts : list of length L
            An alternative past_cl, as similar as possible to the original
            past_cl, that has a possible destination.
        """

        # Find the nearest neighbor
        tree = KDTree(self.centroids)
        dist, ind = tree.query(self.centroids[past_cl[-1],:][None,...],2)
        ind = ind[0,-1]

        # Find the possible pasts of this nearest neighbor
        possible_pasts = []
        for k in self.Q.keys():
            past = list(map(int, k.split(',')))
            if past[-1] == ind:
                possible_pasts.append(past)

        # Assume only one
        return possible_pasts[0]


    def _compute_Q(self):
        """Compute the direct transition matrix of order L."""

        # Initialize the past as the first L centroids
        past_cl = self.cluster_sequence[:self.L].astype(int)
        Q = {}

        for next_cl in self.cluster_sequence[self.L:-1]:

            key = ','.join(map(str, past_cl))

            # Make sure the key exists (and the value is a list)
            if key not in Q:
                Q[key] = []

            Q[key].append(next_cl)

            # update past_cl by removing the oldest element and adding next_cl
            past_cl[:-1] = past_cl[1:]
            past_cl[-1] = next_cl

        # Compute the probabilities
        for k, possible_next in Q.items():

            occurrences = np.bincount(possible_next) / len(possible_next)
            probability = np.empty((0,2),dtype=int)
            for elt in np.unique(possible_next):
                probability = np.vstack((probability,[elt,occurrences[elt]]))

            # Re-write the values with the probabilities
            Q[k] = probability

        return Q

    def _compute_T(self):
        """Compute the transition time"""

        # Number of steps in each sequentially visited cluster
        n_steps_in_cl = np.array([sum(1 for i in g) for k,g in groupby(self.labels)])

        T = {}

        # Loop over
        for i_cl in range(self.cluster_sequence.size-(self.L+1)): # Last transition is neglected

            # Sequential chunks of length self.L+1 (current, next and all pasts)
            cluster_sequence_loc = self.cluster_sequence[i_cl:i_cl+self.L+1]

            transition_time = np.sum(
                    n_steps_in_cl[i_cl+self.L-1:i_cl+self.L+1]
                    )/2. * self.dt

            key = ','.join(map(str, cluster_sequence_loc))

            # Make sure the key exists (and the value is a list)
            if key not in T:
                T[key] = []
            T[key].append(transition_time)

        # Average the transition times of the same sequence of centroids
        total_times = []
        for k, transition_times in T.items():
            T[k] = np.mean(transition_times)
            total_times.append(T[k])

        print('Average transition time: {}'.format(round(np.mean(total_times),3)))

        return T

if __name__=='__main__':

    from sklearn.cluster import KMeans
    np.random.seed(0)

    # CNM config
    K = 5
    L = 3
    dt = 0.016666944449074152

    # get test data and perform clustering (needed for transition properties)
    data = np.load('test_data/data.npy')
    Q_test = np.load('test_data/Q-K{}-L{}.npy'.format(K,L),allow_pickle=True).item()
    T_test = np.load('test_data/T-K{}-L{}.npy'.format(K,L),allow_pickle=True).item()
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=K,max_iter=1000,n_init=100,n_jobs=-1),
            }
    from clustering import Clustering
    clustering = Clustering(**cluster_config)

    # Transition properties
    L = 3
    transition_config = {
            'clustering': clustering,
            'dt': dt,
            'K': K,
            'L': L,
            }
    transition_properties = TransitionProperties(**transition_config)

    # check if the keys of Q are correct
    assert transition_properties.Q.keys() == Q_test.keys()

    # check if proba arrays are the same
    for k in Q_test.keys():
        np.testing.assert_allclose(transition_properties.Q[k], Q_test[k], rtol=1e-2, atol=0)

    # check if the keys of T are correct
    assert transition_properties.T.keys() == T_test.keys()

    # check if T is correct
    assert transition_properties.T == T_test
