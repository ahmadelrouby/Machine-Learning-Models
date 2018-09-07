# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:41:10 2017

"""
import numpy as np

class NearestNeighbor(object):
    #http://cs231n.github.io/classification/
    def __init__(self):
        print ('Nearest Neighbour created')
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, l='L1'):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            print ('Test ', i, ' done.\n')
    
        return Ypred
        
        
        
    def predictKnn(self, X, l='L1',k=1, fold=0):
        """ X is N x D where each row is an example we wish to predict label for """
        print ('K is: ' , k)
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            # find the nearest training example to the i'th test example
            if l == 'L1':
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            else:
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            
            min_values = np.argsort(distances)[:k]
            
            Values = np.zeros((10,), dtype=int)
            
            for value in min_values:
                verdict = self.ytr[value]
                Values[verdict] += 1
            
            leastOne = np.argmax(Values)
            Ypred[i] = leastOne
            print ('Test ', i, ', K: ' , k , ', L: ' , l, ', fold:  ', fold , ' is done.\n')

              
            #print (self.ytr[value])
            #min_index = np.argmin(distances) # get the index with smallest distance
            #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            #print ('Test ', i, ' done.\n')
    
        return Ypred
