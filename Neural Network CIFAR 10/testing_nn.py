#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 23:22:32 2018

@author: ahmadrefaat
"""

import random
import numpy as np
from data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt
from trainer import *
from network import *

data = get_CIFAR10_data(num_training=45000, num_validation=5000, num_test=10000)


"""
51% accuracy for the test set:
learning rate: 1e-3
architecture: 1000,750,500
weight scale: 5e-2
reg: 1e-4    
    
best learning rate: 1e-3
best architecture so far: 500 250 100 250
best reg:


"""


"""
num_train = 50
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}
"""

img_size = 32
start_pixel = 0
cropped_data = {
  'X_train': data['X_train'][:,:,start_pixel:start_pixel+img_size,start_pixel:start_pixel+img_size],
  'y_train': data['y_train'],
  'X_val': data['X_val'][:,:,start_pixel:start_pixel+img_size,start_pixel:start_pixel+img_size],
  'y_val': data['y_val'],
  'X_test': data['X_test'][:,:,start_pixel:start_pixel+img_size,start_pixel:start_pixel+img_size],
  'y_test': data['y_test']
}


print (cropped_data['X_train'].shape)
print (cropped_data['y_train'].shape)


learning_rate = 1e-3  
weight_scale = 5e-2 
reg=1e-5

model = NeuralNet(n_inputs=3*img_size*img_size,
                  n_outputs=10,
                  hidden_dims=[1000,750,400],
                  weight_scale=weight_scale,
                  reg=reg,
                  dropouts=[0.3,0.2,0.1]
                  )

"""model.add_augmentation(rotation_range=10,
                       height_shift_range=0.16,
                       width_shift_range=0.16,
                       img_row_axis=1,
                       img_col_axis=2,
                       img_channel_axis=0,
                       horizontal_flip=True,
                       vertical_flip=False)"""


model.add_augmentation(  rotation_range=10,
                         height_shift_range=0.16,
                         width_shift_range=0.16,
                         img_row_axis=1,
                         img_col_axis=2,
                         img_channel_axis=0,
                         horizontal_flip=False,
                         vertical_flip=False)

#solver = Trainer(model,data,batch_size=200,num_epochs=250,learning_rate=learning_rate,print_every=500)
solver = Trainer(model,cropped_data,batch_size=200,num_epochs=50,
                 optim_configs={'learning_rate':learning_rate},print_every=500)

solver.reset()
solver.train()


plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()

best_model = model
# place missing variables:
X_test, X_val, y_test, y_val = cropped_data['X_test'], cropped_data['X_val'], cropped_data['y_test'], cropped_data['y_val']
y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
print ('Validation set accuracy: ', (y_val_pred == y_val).mean())
print ('Test set accuracy: ', (y_test_pred == y_test).mean())


"""test_acc = solver.check_accuracy(data['X_test'],data['y_test'])
print ("Test Accuracy: " , str(test_acc))
"""