"""
This script implements the generalized self organiwing map with 1-D neighborhoods
(GSOMs with 1DNs) of Gorzałczany and Rudzinski (2018).

The code borrows from sklearn-som 1.1.0 by Riley Smith, accessible here:
    https://pypi.org/project/sklearn-som/ (last accessed Jan. 24, 2023).
    
This just has fit and predict methods, not transform, since it is primarily intended for clustering.

"""

import numpy as np
from numba import jit

class SOM():
    """
    The 1-D, linear self-organizing map class using Numpy.
    """
    def __init__(self, n=2, dim=2, lr=1.0, sigma=1.0, random_state=None, win_min=15, 
                 win_max=20, dcoef=3.0,split_chains=True):
        """
        Parameters
        ----------
        n : int, default=2
            The initial number of neurons in the SOM
        dim : int, default=2
            The dimensionality (number of features) of the input space.
        lr : float, default=1.0
            The initial step size for updating the SOM weights.
        sigma : float, default=1.0
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate). Higher values mean
            more aggressive updates to weights.
        random_state : int, optional
            Optional integer seed to the random number generator for weight
            initialization. This will be used to create a new instance of Numpy's
            default random number generator (it will not call np.random.seed()).
            Specify an integer for deterministic results.
        win_min : int, default = 15
            The minimum number of wins for a neuron, below which a neuron is removed.
        win_max : int, default = 20
            The maximum number of wins for a neuron, above which a new neuron is added.
        dcoef : float, default = 3.0
            The distance coefficient that is multiplied by the average distance 
            when determining whether to disconnect or reconnect chains.
        split_chains : boolean, default = True
            If False, chains are not allowed to be split, and a single continuous chain is fit to the data.
        """
        # Initialize descriptive features of SOM
        self.n = n
        self.dim = dim
        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.win_min = win_min
        self.win_max = win_max
        self.dcoef = dcoef
        self.split_chains = split_chains

        # Initialize weights
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        self.weights = rng.normal(size=(n, dim))
        self.chain = np.zeros(n,dtype=int)
        self.nwin = np.zeros(n,dtype=int)
        
        # Set after fitting
        self._inertia_ = None
        self._n_iter_ = None
        self._trained = False
        self.Q = None

    def _find_bmu(self, x):
        """
        Find the index of the best matching unit for the input vector x.
        """
        # Stack x to have one row per weight
        x_stack = np.stack([x]*(self.n), axis=0)
        # Calculate distance between x and each weight
        distance = np.linalg.norm(x_stack - self.weights, axis=1)
        # Find index of best matching unit
        return np.argmin(distance)

    @staticmethod
    @jit(nopython=True)
    def train(data,n,dim,lr,lr_change,sigma,weights,chain):
        """
        Do one epoch of training on all the data.
        This is made as a static method so that it can be made faster with numba.
        
        Parameters
        ----------
        data : ndarray, float
            Training data. Must have shape (N, dim) where N is the number
            of training samples.
        n : int
            The initial number of neurons in the SOM
        dim : int
            The dimensionality (number of features) of the input space.
        lr : float
            The initial step size for updating the SOM weights.
        sigma : ndarray, float
            Controls the magnitude of change to each weight.
        weights : ndarray
            The neuron weights. Has size (n,dim)
        chain : ndarray, int
            The index of the chain to which each neuron belongs. Has size (n).

        Returns
        -------
        None
        """
        nwin = np.zeros(n,dtype=np.int32)
        indices = np.arange(data.shape[0])
        for idx in indices:
            # Do one step of training
            x = data[idx,:]

            # Get index of best matching unit (same way as in the _find_bmu function)
            distance = np.sqrt(np.sum((x - weights)**2,axis=1))
            bmu_index = np.argmin(distance)

            # Find square distance from each weight to the BMU
            all_inds = np.arange(n,dtype=np.float64)
            bmu_distance = np.power(all_inds - bmu_index, 2.0)

            # Compute update neighborhood
            neighborhood = np.exp((bmu_distance / (2 * sigma ** 2)) * -1) #As given in Eqn. 5 of Gorzalczany and Rudzinski (2018).
            local_step = lr * neighborhood
            
            #Set the neighborhood influence to 0 for points not in the same chain.
            local_step[chain != chain[bmu_index]] = 0

            # Multiply by difference between input and weights
            delta = (local_step*(x-weights).T).T #Using stack on local_step wasn't working with Numba, but this does.
            
            # Update weights
            weights += delta
            
            # Update the count of wins.
            nwin[bmu_index] += 1
                    
            # Update learning rate
            lr = lr + lr_change
        
        return weights, nwin, lr
    
    def remove_neurons(self,to_remove):
        """

        Parameters
        ----------
        to_remove : boolean array
            Array of size self.n, which is true for neurons to be removed and false otherwise.

        Returns
        -------
        None.

        """
        if any(to_remove):
            chain_nums = np.unique(self.chain) #Get these before removing neurons in case any chains are removed entirely.
            self.weights = self.weights[~to_remove]
            self.chain = self.chain[~to_remove]
            self.nwin = self.nwin[~to_remove]
            self.n -= np.sum(to_remove)
            for i in range(chain_nums.size):
                #If any chains have been removed entirely, move other chain numbers down.
                if not any(self.chain==chain_nums[i]):
                    self.chain[self.chain>chain_nums[i]] -= 1
                    chain_nums[i+1:] -= 1
        
    
    def insert_neuron(self,ind,weight,chain):
        """

        Parameters
        ----------
        ind : int
            Neuron after which to insert the new neuron
        weight : float
            Weight of the new neuron
        chain : int
            Chain to which the new neuron belongs

        Returns
        -------
        None.

        """
        
        self.weights = np.concatenate((self.weights[0:ind+1],weight,self.weights[ind+1:]))
        self.chain = np.concatenate((self.chain[0:ind+1],[chain],self.chain[ind+1:]))
        self.nwin = np.concatenate((self.nwin[0:ind+1],[0],self.nwin[ind+1:]))
        self.n += 1
    
    def move_neurons(self,ind1,ind2,ind_new,flip):
        """
        Move a group of neurons to another part of the chain.
        Note: This is primarily for moving forward in the chain. If moving back, 
        it will be necessary to choose ind_new for the chain excluding the part being moved.

        Parameters
        ----------
        ind1 : int
            Index of the first neuron in the group to be moved.
        ind2 : int
            Index of the last neuron in the group to be moved.
        ind_new : int
            Index to move the first nuron in the group to.
        flip : boolean
            Whether or not to flip the order of the neurons being moved

        Returns
        -------
        None.

        """
        w = np.concatenate((self.weights[0:ind1],self.weights[ind2+1:])) #The weights not being moved.
        c = np.concatenate((self.chain[0:ind1],self.chain[ind2+1:]))
        nw = np.concatenate((self.nwin[0:ind1],self.nwin[ind2+1:]))
        if not flip:
            self.weights = np.concatenate((w[0:ind_new],self.weights[ind1:ind2+1],w[ind_new:]))
            self.chain = np.concatenate((c[0:ind_new],self.chain[ind1:ind2+1],c[ind_new:]))
            self.nwin = np.concatenate((nw[0:ind_new],self.nwin[ind1:ind2+1],nw[ind_new:]))
        else:
            self.weights = np.concatenate((w[0:ind_new],self.weights[ind2:ind1-1:-1],w[ind_new:]))
            self.chain = np.concatenate((c[0:ind_new],self.chain[ind2:ind1-1:-1],c[ind_new:]))
            self.nwin = np.concatenate((nw[0:ind_new],self.nwin[ind2:ind1-1:-1],nw[ind_new:]))
        
    
    def update_chains(self):
        """
        Update the number and connection of chains using the 4 successive 
        operations described by Gorzalczany and Rudzinski (2018).

        Returns
        -------
        None.
        """
        
        # Condition 1: Remove low-activity neurons
        low_act = self.nwin < self.win_min
        if self.n-np.count_nonzero(low_act) >= 2: #There always need to be at least 2 neurons in the model, or there will be no chains.
            self.remove_neurons(low_act)
            
        # Condition 2: Disconnect chains and remove single-neuron subchains
        if self.split_chains:
            d = np.concatenate((np.linalg.norm(self.weights[1:] - self.weights[0:-1], axis=1),[0])) #Euclidean distance between adjacent neurons.
            to_remove = np.zeros(self.n,dtype=bool)
            for j in range(self.n):
                m = np.sum(self.chain==self.chain[j])
                if (m >= 3) and (j<self.n-1) and ((self.chain[j]==self.chain[j+1])):
                    mask = (self.chain==self.chain[j])
                    mask[j] = False
                    davg = np.sum(d[mask][:-1])/(m-2.0)
                    if d[j] > self.dcoef*davg:
                        self.chain[self.chain>self.chain[j]] += 1 #First shift other chain numbers up to make room for a new one.
                        mask[0:j] = False
                        self.chain[mask] += 1 #Split the chain.
                    m = np.sum(self.chain==self.chain[j]) #Recalculate this so if it's now 1, it will be removed below.
                if m == 1:
                    to_remove[j] = True #Remove single neuron chain.
            self.remove_neurons(to_remove)
            
        # Condition 3a: Insert new neurons between any two adjacent high-activity neurons
        high_act = self.nwin > self.win_max
        ind_insert = np.nonzero(high_act[0:-1] & high_act[1:] & (self.chain[0:-1] == self.chain[1:]))[0]
        for i in range(ind_insert.size):
            self.insert_neuron(ind_insert[i],np.mean(self.weights[ind_insert[i]:ind_insert[i]+2],axis=0,keepdims=True),
                               self.chain[ind_insert[i]])
            ind_insert[i+1:] += 1 #Since we've added a neuron, the indices of the rest increase.
        
        
        # Condition 3b: Insert a new neuron into the neighborhood of an end-chain high-activity neuron
        chain_nums = np.unique(self.chain)
        for i in range(chain_nums.size):
            end_inds = np.squeeze(np.argwhere(self.chain==chain_nums[i])[[0,-1]])
            if self.nwin[end_inds[0]] > self.win_max:
                self.insert_neuron(end_inds[0]-1,np.expand_dims(2*self.weights[end_inds[0]]-self.weights[end_inds[0]+1],0),chain_nums[i])
            if self.nwin[end_inds[1]] > self.win_max:
                self.insert_neuron(end_inds[1],np.expand_dims(2*self.weights[end_inds[1]]-self.weights[end_inds[1]-1],0),chain_nums[i])
                
        # Condition 4: Reconnect chains
        # I'm not sure if only adjacent chains can be reconnected or if any chain can be connected to any chain.
        # Now, I'm allowing any chain to connect to any other chains.
        if self.split_chains:
            d = np.concatenate((np.linalg.norm(self.weights[1:] - self.weights[0:-1], axis=1),[0])) #Euclidean distance between adjacent neurons.
            chain_nums = np.unique(self.chain)
            for i in range(chain_nums.size-1):
                for j in range(i+1,chain_nums.size):
                    if any(self.chain==chain_nums[i]) and any(self.chain==chain_nums[j]): #It's possible that this chain was already added to another one.
                        inds1 = np.squeeze(np.argwhere(self.chain==chain_nums[i]))[[0,-1]] #End points of this chain.
                        inds2 = np.squeeze(np.argwhere(self.chain==chain_nums[j]))[[0,-1]]
                        #Find which of the end points of each line are the closest to each other.
                        d_eS1_eS2 = 0
                        for ii in range(2):
                            for jj in range(2):
                                dij = np.linalg.norm(self.weights[inds1[ii],:] - self.weights[inds2[jj],:])
                                if (ii==0 and jj==0) or (dij<d_eS1_eS2):
                                    d_eS1_eS2 = dij
                                    ind1 = inds1[ii] #Index of end point to use from this chain.
                                    ind2 = inds2[jj] #Index of end point to use from the next chain.
                        mS1 = np.sum(self.chain==chain_nums[i]) #m for this chain
                        mS2 = np.sum(self.chain==chain_nums[j]) #m for the next chain
                        davgS1 = np.sum(d[self.chain==chain_nums[i]][:-1])/(mS1-1.0)
                        davgS2 = np.sum(d[self.chain==chain_nums[j]][:-1])/(mS2-1.0)
                        if d_eS1_eS2 < self.dcoef*min(davgS1,davgS2):
                            if ind1 == inds1[0]:
                                if ind2 == inds2[0]:
                                    #Both 1st points. Move 2nd chain in front of first and flip it.
                                    self.move_neurons(inds2[0],inds2[1],ind1,True)
                                else:
                                    #1st point of 1st chain, end point of 2nd chain. Move second chain in front of first.
                                    self.move_neurons(inds2[0],inds2[1],ind1,False)
                            else:
                                if ind2 == inds2[0]:
                                    #End point of 1st chain, 1st point of 2nd chain. Move second chain after first.
                                    self.move_neurons(inds2[0],inds2[1],ind1,False)
                                else:
                                    #Both end points. Move 2nd chain after first and flip it.
                                    self.move_neurons(inds2[0],inds2[1],ind1,True)
                            self.chain[(self.chain==chain_nums[j])] = chain_nums[i]
                            #Because we're moving things around within the chain, we also need to update d.
                            d = np.concatenate((np.linalg.norm(self.weights[1:] - self.weights[0:-1], axis=1),[0]))
            #Reduce chain numbers so there are not any skipped numbers.
            chain_nums = np.unique(self.chain)
            for i in range(chain_nums.size):
                while (i==0 and chain_nums[i]!=0) or (chain_nums[i]-chain_nums[i-1]>1):
                    #Shift this and all higher chain numbers down by 1.
                    self.chain[self.chain>=chain_nums[i]] -= 1 
                    chain_nums[i:] -= 1

    def _compute_point_intertia(self, x):
        """
        Compute the inertia of a single point. Inertia defined as squared distance
        from point to closest cluster center (BMU)
        """
        # Find BMU
        bmu_index = self._find_bmu(x)
        bmu = self.weights[bmu_index]
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))

    def fit(self, X, epochs=1000, shuffle=True):
        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.

        Parameters
        ----------
        X : ndarray
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        epochs : int, default=1
            The number of times to loop through the training data when fitting.
        shuffle : bool, default True
            Whether or not to randomize the order of train data when fitting.
            Can be seeded with np.random.seed() prior to calling fit.

        Returns
        -------
        None
            Fits the SOM to the given data but does not return anything.
        """
        # Count total number of iterations
        global_iter_counter = 0
        n_samples = X.shape[0]
        total_iterations = epochs * n_samples
        
        #Change of learning rate with each iteration.
        lr_change = (-1.0 / total_iterations) * self.initial_lr
        
        self.Q = np.zeros(epochs)
        for epoch in range(epochs):

            if shuffle:
                rng = np.random.default_rng(self.random_state)
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            # Train
            w_prev = self.weights.copy()
            weights, nwin, lr = self.train(X[indices,:],self.n,self.dim,self.lr,lr_change,self.sigma,self.weights,self.chain)
            self.weights = weights
            self.nwin = nwin
            self.lr = lr
            global_iter_counter += n_samples
                
            #Compute Q, the criterion for weights stabilizing in Eqn. 6 of Gorzalczany and Rudzinski (2018).
            self.Q[epoch] = np.sum(np.sqrt(np.sum((self.weights-w_prev)**2,axis=1)))
                
            # Update number of neurons and chain connections
            self.update_chains()
            

        # Compute inertia
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        self._inertia_ = inertia

        # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

        # Set trained flag
        self._trained = True

        return

    def predict(self, X):
        """
        Predict cluster for each element in X.

        Parameters
        ----------
        X : ndarray
            An ndarray of shape (n, self.dim) where n is the number of samples.
            The data to predict clusters for.

        Returns
        -------
        labels : ndarray
            An ndarray of shape (n,). The predicted cluster index for each item
            in X.
        """
        # Check to make sure SOM has been fit
        if not self._trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'

        neuron_ind = np.array([self._find_bmu(x) for x in X])
        labels = self.chain[neuron_ind]
        return labels

    def fit_predict(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by predict(X).

        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim). The data to fit and then predict.
        **kwargs
            Optional keyword arguments for the .fit() method.

        Returns
        -------
        labels : ndarray
            ndarray of shape (n,). The index of the predicted cluster for each
            item in X (after fitting the SOM to the data in X).
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)

    @property
    def cluster_centers_(self):
        return self.weights

    @property
    def inertia_(self):
        if self._inertia_ is None:
            raise AttributeError('SOM does not have inertia until after calling fit()')
        return self._inertia_

    @property
    def n_iter_(self):
        if self._n_iter_ is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit()')
        return self._n_iter_
