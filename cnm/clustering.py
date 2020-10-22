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

import numpy as np
import os

class Clustering:
    """Perform the data clustering with the requested clustering algorithm.

    Attributes
    ----------
    labels : ndarray of shape (n_snapshots,)
        Cluster affiliation of each snapshot.
    centroids : ndarray of shape (K,n_dim)
        Centroids of the clusters.
    cluster_sequence : ndarray of shape (# transition+1,)
        Sequence of visited clusters.
    """

    def __init__(self,data,cluster_algo,dataset):
        """
        Parameters
        ----------
        data : ndarray of shape (n_snapshots,n_dim)
            Snapshots of the dynamical system, equally spaced in time.
        cluster_algo : object
            Instance from the selected clustering class. Must provide a 'fit()'
            method and return labels_ and cluster_centers_.
        dataset : str
            A label defining the dataset (e.g., 'lorenz', 'boundary_layer',
            ...). Defines the folder where the clustering output will be
            stored.
        """

        # Perform clustering
        print('Perform clustering')
        print('------------------')
        print('Use {} clusters'.format(cluster_algo.n_clusters))

        # Set seeding to reproduce the same results
        np.random.seed(0)

        # Ouput path (create folder if necessary)
        data_folder = 'output/{}'.format(dataset)
        data_path = os.path.join(
                data_folder,'clustering-K{}'.format(cluster_algo.n_clusters)
                )
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        if not os.path.exists(data_path+'.npz'):
            print('Compute and save in {}'.format(data_path+'.npz'))

            cluster_algo.fit(data)

            self.labels = cluster_algo.labels_
            self.centroids = cluster_algo.cluster_centers_
            diff = np.diff(self.labels)
            self.cluster_sequence = self.labels[np.insert(diff.astype(np.bool), 0, True)]

            np.savez(
                    data_path,
                    labels = cluster_algo.labels_,
                    centroids = cluster_algo.cluster_centers_,
                    cluster_sequence = self.cluster_sequence
                    )

        else:
            print('Read from {}'.format(data_path+'.npz'))
            data = np.load(
                    data_path+'.npz',
                    )
            self.labels = data['labels']
            self.centroids = data['centroids']
            self.cluster_sequence = data['cluster_sequence']
        print('\n')

if __name__=='__main__':

    from sklearn.cluster import KMeans
    import numpy as np

    # number of clusters
    k = 5

    # get test data
    data = np.load('test_data/data.npy')
    centroids_test = np.loadtxt('test_data/centroids-K{}'.format(k))
    labels_test = np.loadtxt('test_data/labels-K5'.format(k))

    # perform clustering
    cluster_config = {
            'data': data,
            'cluster_algo': KMeans(n_clusters=k,max_iter=1000,n_init=100,n_jobs=-1),
            'dataset': 'dummy',
            }
    clustering = Clustering(**cluster_config)

    # check clustering
    assert np.all(clustering.centroids == centroids_test)

    # check labels
    assert np.all(clustering.labels == labels_test)
