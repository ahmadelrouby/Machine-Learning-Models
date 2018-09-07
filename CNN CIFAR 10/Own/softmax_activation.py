#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 00:57:28 2018

@author: ahmadrefaat
"""
import numpy as np
class Softmax:
    def __init__(self):
        self.name = "Softmax Activation"
        self.type = "Loss"
        self.trainable_params = False
        
    def forward_backward(self,x,y,mode="training"):
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx

    