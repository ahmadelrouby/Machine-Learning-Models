#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 00:43:23 2018

@author: ahmadrefaat
"""

import numpy as np
class Flatten:
    def __init__(self):
        self.name = "Flatten Layer"
        self.cache_input = None
        self.trainable_params = False
    
    def forward(self,X,mode="training"):
         output_sum = np.product(X.shape[1:])
         self.cache_input = X.shape
         return np.reshape(X,(X.shape[0],output_sum))
         
    
    def backward(self,dout,mode="training"):
        return np.reshape(dout,(self.cache_input))
    
    
    
    

"""arr = np.zeros((1000,3,32,32))
myFlat = flatten()

fpass = myFlat.forward(arr)

b = np.zeros((1000,3*32*32))
bpass = myFlat.backward(b)

print (fpass.shape)
print (bpass.shape)
"""        