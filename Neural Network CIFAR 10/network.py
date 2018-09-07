#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 22:19:30 2018

@author: ahmadrefaat
"""

import numpy as np
from utils import *
from data_utils import *

class NeuralNet(object):
    
    
    def __init__(self,n_inputs,n_outputs,hidden_dims,weight_scale,reg,dropouts=None,batch_norm=False):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.num_layers = 1 + len(hidden_dims)
        self.ws = weight_scale
        self.reg = reg
        self.params = {}
        self.augment = False
      
        self.dropouts = dropouts
        self.batch_norm = batch_norm
        
        self.use_drop = True
        
        if dropouts is None:
            self.use_drop = False
     
        
        
        dims = [n_inputs] + hidden_dims + [n_outputs]
        
        for i in range(self.num_layers): 
            self.params['b%d' % (i+1)] = np.zeros(dims[i + 1])
            self.params['W%d' % (i+1)] = np.random.randn(dims[i], dims[i + 1])/np.sqrt(dims[i]/2.0)# * weight_scale
            
            
        if self.batch_norm:
            self.bn_params = {'bn_param' + str(i + 1): {'mode': 'train',
                                                        'running_mean': np.zeros(dims[i + 1]),
                                                        'running_var': np.zeros(dims[i + 1])}
                              for i in range(len(dims) - 2)}
            gammas = {'gamma' + str(i + 1):
                      np.ones(dims[i + 1]) for i in range(len(dims) - 2)}
            betas = {'beta' + str(i + 1): np.zeros(dims[i + 1])
                     for i in range(len(dims) - 2)}

            self.params.update(betas)
            self.params.update(gammas)
                
    def add_augmentation(self,rotation_range=0,
                         height_shift_range=0,
                         width_shift_range=0,
                         img_row_axis=1,
                         img_col_axis=2,
                         img_channel_axis=0,
                         horizontal_flip=False,
                         vertical_flip=False):
        
        self.rotation_range = float(rotation_range)
        self.height_shift_range = float(height_shift_range)
        self.width_shift_range= float(width_shift_range)
        self.img_row_axis= int(img_row_axis)
        self.img_col_axis= int(img_col_axis)
        self.img_channel_axis= int(img_channel_axis)
        self.horizontal_flip= bool(horizontal_flip)
        self.vertical_flip= bool(vertical_flip)
        self.augment = True
        
        
    def loss(self,X,y=None):
        
        
        if y is None:
            mode = 'test'
        else:
            mode = 'train'

        if self.batch_norm:
            for key, bn_param in self.bn_params.items():
                bn_param['mode'] = mode
                
                
        if self.augment and mode == 'train':
            X = augment_batch(X,
                              rotation_range=self.rotation_range,
                              height_shift_range=self.rotation_range,
                              width_shift_range=self.width_shift_range,
                              img_row_axis=self.img_row_axis,
                              img_col_axis=self.img_col_axis,
                              img_channel_axis=self.img_channel_axis,
                              horizontal_flip=self.horizontal_flip,
                              vertical_flip=self.vertical_flip)
        
        
            
            
        X = np.reshape(X, (X.shape[0], -1))

        layer = {}
        drouput_layer = {}
        drouput_cache = {}
        
        
        loss, grads = 0.0, {}
        
        layer[0] = X
        drouput_layer[0] = X
        
        cache_layer = {}
        for i in range(1, self.num_layers):
            
            if self.batch_norm:
                gamma = self.params['gamma' + str(i)]
                beta = self.params['beta' + str(i)]
                bn_param = self.bn_params['bn_param' + str(i)]
                
                if self.use_drop:
                    layer[i], cache_layer[i] = norm_relu_forward(drouput_layer[i - 1],
                                                                    self.params['W%d' % i],
                                                                    self.params['b%d' % i], 
                                                                    gamma, beta, bn_param)
                    drouput_layer[i], drouput_cache[i] = dropout_forward(layer[i],self.dropouts[i-1],mode)
                else:
                    layer[i], cache_layer[i] = norm_relu_forward(layer[i - 1],
                                                                    self.params['W%d' % i],
                                                                    self.params['b%d' % i], 
                                                                    gamma, beta, bn_param)
            else:
                if self.use_drop:
                    layer[i], cache_layer[i] = forward_layer_then_relu(drouput_layer[i - 1],
                                                                       self.params['W%d' % i],
                                                                       self.params['b%d' % i]) 
                    
                    drouput_layer[i], drouput_cache[i] = dropout_forward(layer[i],self.dropouts[i-1],mode)
                else:
                    layer[i], cache_layer[i] = forward_layer_then_relu(layer[i - 1],self.params['W%d' % i],self.params['b%d' % i])   
                
            
            
            
            
        last_W = 'W%d' % self.num_layers
        last_b = 'b%d' % self.num_layers
            
        if self.use_drop:
            scores, old = forward_layer(drouput_layer[self.num_layers - 1],self.params[last_W],self.params[last_b])
        else:
            scores, old = forward_layer(layer[self.num_layers - 1],self.params[last_W],self.params[last_b])
        
        if y is None:
            return scores
        
        loss, scores_grad = softmax_loss_gradient(scores,y)
        
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params['W%d' % i]**2)
        
        
        dx = {}
        grads[last_W], grads[last_b],  dx[self.num_layers] = backward_layer(scores_grad, old)
        grads[last_W] += self.reg * self.params[last_W]
        
        
        
        for i in reversed(range(1, self.num_layers)):
            if self.batch_norm:
                if self.use_drop:
                    drop_grad = dropout_backward(dx[i+1],drouput_cache[i],mode)
                    dW, dB, dxA, dGamma, dBeta = norm_relu_backward(drop_grad, cache_layer[i])   
                    grads['W%d' % i] = dW 
                    grads['b%d' % i] = dB
                    dx[i] = dxA 
                    grads['gamma%d' % i] = dGamma 
                    grads['beta%d' % i] = dBeta 
                else:
                    dW, dB, dxA, dGamma, dBeta = norm_relu_backward(dx[i + 1], cache_layer[i])  
                    grads['W%d' % i] = dW 
                    grads['b%d' % i] = dB
                    dx[i] = dxA 
                    grads['gamma%d' % i] = dGamma 
                    grads['beta%d' % i] = dBeta                    
            else:
                if self.use_drop:
                    drop_grad = dropout_backward(dx[i+1],drouput_cache[i],mode)
                    grads['W%d' % i], grads['b%d' % i], dx[i] = backward_layer_then_relu(drop_grad, cache_layer[i])
                    grads['W%d' % i] += self.reg * self.params['W%d' % i]
                else:
                    grads['W%d' % i], grads['b%d' % i], dx[i] = backward_layer_then_relu(dx[i + 1], cache_layer[i])
                    grads['W%d' % i] += self.reg * self.params['W%d' % i]
        
        return loss, grads
        
        
        
        
        