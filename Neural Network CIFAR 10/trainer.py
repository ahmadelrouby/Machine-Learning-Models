#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 23:03:54 2018

@author: ahmadrefaat
"""

from utils import *
import numpy as np


import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


class Trainer(object):
    def __init__(self,model,data,batch_size=200,num_epochs=100,learning_rate=1e-3,verbose=True,print_every=50,optim_configs={}):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.print_every = print_every
        self.verbose = verbose
        self.optim_config = optim_configs
    
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
          mask = np.random.choice(N, num_samples)
          N = num_samples
          X = X[mask]
          y = y[mask]
    
        # Compute predictions in batches
        num_batches = int(N / batch_size)
        if N % batch_size != 0:
          num_batches += 1
        y_pred = []
        for i in range(num_batches):
          start = i * batch_size
          end = (i + 1) * batch_size
          scores = self.model.loss(X[start:end])
          y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
    
        return acc


    def step(self):
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]
        
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
        
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = adam(w,dw,config)
            #next_w = sgd(w, dw, self.learning_rate)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config
        
    def reset(self):
       
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)
        training_loss = 0
        training_loss_sum = 0 
        training_num = 0
        self.actual_training_loss = []
        
        for t in range(num_iterations):
            self.step()
            training_num += 1
            training_loss_sum += self.loss_history[len(self.loss_history)-1]
            training_loss = float(training_loss_sum)/(training_num)
            
            #if t % self.print_every == 0:
                #print('(Iteration %d / %d) loss: %f'% (t + 1, num_iterations, self.loss_history[-1]))
            
            progress(int(t-(iterations_per_epoch*self.epoch)), iterations_per_epoch, status='Loss: ' + 
                     str(training_loss))
            
            
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                training_num = 0
                training_loss_sum = 0
                self.actual_training_loss.append(training_loss)
                
            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                
                train_acc = self.check_accuracy(self.X_train, self.y_train,num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                
                if self.verbose:
                    print ('(Epoch %d / %d) train acc: %f; val_acc: %f' % (self.epoch, self.num_epochs, train_acc, val_acc))

                # Keep track of the best model
                if val_acc > self.best_val_acc:
                  self.best_val_acc = val_acc
                  self.best_params = {}
                  for k, v in self.model.params.items():
                    self.best_params[k] = v.copy()
                        
        self.model.params = self.best_params
        
            
            
    