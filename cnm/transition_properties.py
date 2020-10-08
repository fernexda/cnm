# -*- coding: utf-8 -*-

import numpy as np

class TransitionProperties:

    def __init__(self, clustering, K: int, L: int, dt):

        self.labels = clustering.labels
        self.K = K
        self.L = L
        self.dt = dt

        # Safety check
        if self.L <= 0:
            raise Exception('The model order must be > 0')

        self.compute_Q()

    def compute_Q(self):
        """Compute the direct transition matrix of order L."""

        # Get the sequence of visited clusters
        diff = np.diff(self.labels)
        # idx_transition give the first index of a new cluster
        idx_transition = np.where(diff!=0)[0] + 1
        Idx_transition = np.insert(idx_transition,0,0)
        cluster_sequence = self.labels[np.insert(diff.astype(np.bool), 0, True)]

        # Initialize the past as the first L centroids
        past_cl = cluster_sequence[:self.L]
        self.Q = {}

        for next_cl in cluster_sequence[self.L:-2]:

            key = ','.join(map(str, past_cl))

            if key not in self.Q:
                self.Q[key] = []

            self.Q[key].append(next_cl)

            # update past_cl by removing the oldest element and adding next_cl
            past_cl[:-1] = past_cl[1:]
            past_cl[-1] = next_cl

        # Compute the probabilities
        for k, possible_next in self.Q.items():

            BC = np.bincount(possible_next) / len(possible_next)
            probability = np.empty((0,2),dtype=int)
            for elt in np.unique(possible_next):
                probability = np.vstack((probability,[elt,BC[elt]]))
        
            # --> Re-write the probabilities
            self.Q[k] = probability

        for k,v in self.Q.items():
            print(k)
            for i in range(v.shape[0]):
                print('  ',v[i,0],v[i,1])
        exit()
#        
#            # --> Get the values
#            Values = QNested.getValues(kGal)
#        
#            # --> Compute the corresponding probabilities
#            BC = np.bincount(Values) / Values.size
#            Proba = np.empty((0,2),dtype=int)
#            for elt in np.unique(Values):
#                Proba = np.vstack((Proba,[elt,BC[elt]]))
#        
#            # --> Re-write the probabilities
#            QNested.addToDict('add',kGal,Proba)
#        
#        # --> Convert from default dict to normal dict
#        QNested.DDict2Dict()
#





        print(idx_start)
        print(past)
        print('labels')
        for i in range(idx_start+5):
            print(self.labels[i])
        print('labels[idx_start]',self.labels[idx_start])

        print(cluster_sequence[:self.L+1])



## --------------------------------------------------------------------------------
#        labels = self.labels
#        # --> general configuration
#        n = self.npast
#        qnested = nesteddict()
#
#        diff = np.diff(labels)
#        # --> idxtransition give the first index of a new cluster
#        idxtransition = np.where(diff!=0)[0] + 1
#        idxtransition = np.insert(idxtransition,0, 0)
#        
#        clusternumbers = labels[np.insert(diff.astype(np.bool), 0, true)]
#        
#        # --> get past. n=0 means no past.
#        past = clusternumbers[:n]
#        idxstart = np.where(diff != 0)[0][n] # start just before next transition
#        keysgal = []
#        
#        # --> start checking the transitions
#        # note: we neglect the last transition to keep exactly the same
#        #       transitions as those identified by tau. otherwise, cnm can look
#        #       in taunested for keys that don't exist.
#        for i in range(idxstart,idxtransition[-2]):
#        
#            # --> transition detected (between i+n and i+n-1)
#            if labels[i+1] != labels[i]:
#        
#                # --> get dictionary keys:
#                keys = list(Past)
#                Keys.append(Labels[i])
#        
#                # --> Value:
#                Value = Labels[i+1]
#        
#                # --> Check if keys already exist:
#                if not QNested.checkExistence(Keys):
#                    QNested.addToDict('add',Keys,[])
#        
#                # --> Append value
#                QNested.addToDict('append',Keys,Value)
#        
#                # --> Update the past
#                Past = Keys[1:]
#        
#                # --> Keep memory of the keys
#                if not Keys in KeysGal:
#                    KeysGal.append(Keys)
#
#
#
#
#        # --> Get the transition probabilities
#        for kGal in KeysGal:
#        
#            # --> Get the values
#            Values = QNested.getValues(kGal)
#        
#            # --> Compute the corresponding probabilities
#            BC = np.bincount(Values) / Values.size
#            Proba = np.empty((0,2),dtype=int)
#            for elt in np.unique(Values):
#                Proba = np.vstack((Proba,[elt,BC[elt]]))
#        
#            # --> Re-write the probabilities
#            QNested.addToDict('add',kGal,Proba)
#        
#        # --> Convert from default dict to normal dict
#        QNested.DDict2Dict()
#
