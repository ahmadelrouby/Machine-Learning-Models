#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 22:08:33 2018

@author: ahmadrefaat
"""

import numpy as np


def sgd(w, dw, learning_rate):    
    w -= learning_rate * dw
    return w



def adam(x, dx, config=None):
 
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
 
    
  learning_rate, beta1, beta2, eps, m, v, t \
    = config['learning_rate'], config['beta1'], config['beta2'], \
      config['epsilon'], config['m'], config['v'], config['t']

  t += 1
  m = beta1 * m
  temp = (1 - beta1) * dx
  m = m + temp
  
  v = beta2 * v + (1 - beta2) * (dx**2)

  # bias correction:
  mb = m / (1 - beta1**t)
  vb = v / (1 - beta2**t)

  next_x = -learning_rate * mb / (np.sqrt(vb) + eps) + x

  config['m'], config['v'], config['t'] = m, v, t

  return next_x, config



def forward_layer(X,weights,bias):
    output = X.reshape(X.shape[0], weights.shape[0]).dot(weights) + bias
    old = (X,weights,bias)
    return output,old



def backward_layer(output, old):   
    X,weights,bias = old
    dw = X.reshape(X.shape[0], weights.shape[0]).T.dot(output)
    db = np.sum(output, axis=0)
    dx = output.dot(weights.T).reshape(X.shape)
    return dw,db,dx



def forward_relu(inp):
    return np.maximum(0,inp) , inp




def backward_relu(output, old):  
    grad = output
    grad[old < 0] = 0
    return grad


def dropout_forward(inp, prob, mode):  
    if mode == 'train':
        value = np.random.rand(*inp.shape)     
        mask = ((value > prob))
        out = inp * mask
    elif mode == 'test':
        mask = None
        out = inp
        
    return out, mask

def dropout_backward(output, mask, mode):
    if mode == 'train':
        grad = output * mask
    elif mode == 'test':
        grad = output
   
    return grad
    

def forward_layer_then_relu(x,w,b):
    output1,old1 = forward_layer(x,w,b)
    output2,old2 = forward_relu(output1)
    old = (old1,old2)
    return output2, old

def backward_layer_then_relu(output,old):
    layer_old , relu_old = old
    relu_grad = backward_relu(output,relu_old)
    dw,db,dx = backward_layer(relu_grad,layer_old)
    return dw,db,dx


def softmax_loss_gradient(x,y):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
    
def batchnorm_forward(x, gamma, beta, bn_param):
   
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':   
        mu = 1 / float(N) * np.sum(x, axis=0)
        xmu = x - mu
        carre = xmu**2
        var = 1 / float(N) * np.sum(carre, axis=0)
        sqrtvar = np.sqrt(var + eps)
        invvar = 1. / sqrtvar
        va2 = xmu * invvar
        va3 = gamma * va2
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, carre, var, sqrtvar, invvar,
                 va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':
        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum((x - mu) * (var + eps)**(-1. / 2.) * dout, axis=0)
    dx = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dout - np.sum(dout, axis=0)
                                                       - (x - mu) * (var + eps)**(-1.0) * np.sum(dout * (x - mu), axis=0))

    return dx, dgamma, dbeta


def norm_relu_forward(x, w, b, gamma, beta, bn_param):
    h, h_cache = forward_layer(x, w, b)
    hnorm, hnorm_cache = batchnorm_forward(h, gamma, beta, bn_param)
    hnormrelu, relu_cache = forward_relu(hnorm)
    cache = (h_cache, hnorm_cache, relu_cache)

    return hnormrelu, cache


def norm_relu_backward(dout, cache):
    h_cache, hnorm_cache, relu_cache = cache

    dhnormrelu = backward_relu(dout, relu_cache)
    dhnorm, dgamma, dbeta = batchnorm_backward(dhnormrelu, hnorm_cache)
    dw, db, dx = backward_layer(dhnorm, h_cache)

    return dw, db, dx, dgamma, dbeta

