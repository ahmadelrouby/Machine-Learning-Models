import numpy as np
from numpy.linalg import inv

class CIFAR_10_CLASSIFIER(object):

    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y
        
        size = self.Xtr.shape[0]
        ones = np.ones((size,1)) 
        self.Xtr = np.hstack((self.Xtr,ones))
         
        #print (self.Xtr)
        x_transpose = self.Xtr.T
        
        
        for i in range(10):
            new_y = (self.ytr == i)
            w = ((inv((x_transpose).dot(self.Xtr))).dot(x_transpose)).dot(new_y)
            if(i == 0):
                self.all_ws = w
            else:
                self.all_ws = np.vstack((self.all_ws,w))
            

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        
        size = X.shape[0]
        ones = np.ones((size,1)) 
        X = np.hstack((X,ones))
        
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        for i in range(num_test):
            
            results = (self.all_ws).dot(X[i,:])
            max_index = np.argmax(results)
            Ypred[i] = max_index

        return Ypred
        
        
    