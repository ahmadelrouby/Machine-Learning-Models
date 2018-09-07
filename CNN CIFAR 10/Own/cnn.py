#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 01:48:04 2018

@author: ahmadrefaat
"""


from conv_layer import ConvolutionalLayer
from dense import Dense
from max_pool_layer import MaxPool
from relu_activation import Relu
from softmax_activation import Softmax
from flatten import Flatten
from model import Model
from data_utils import *



data = get_CIFAR10_data()

model = Model(data['X_train'],data['y_train'],data['X_test'],data['y_test'],num_epochs=50,batch_size = 250)


model.add_augmentation(  rotation_range=5,
                         height_shift_range=0.16,
                         width_shift_range=0.16,
                         img_row_axis=1,
                         img_col_axis=2,
                         img_channel_axis=0,
                         horizontal_flip=True,
                         vertical_flip=False)



model.add_layer(ConvolutionalLayer(num_filters=32))
model.add_layer(Relu())

model.add_layer(ConvolutionalLayer(input_shape=[32,32,32],num_filters=32,filter_dims=[32,3,3]))
model.add_layer(Relu())

model.add_layer(MaxPool())

model.add_layer(Flatten())

model.add_layer(Dense(input_shape=8192,neurons=650))
model.add_layer(Relu())


#model.add_layer(Dense(input_shape=1000,neurons=650))
#model.add_layer(Relu())


model.add_layer(Dense(input_shape=650,neurons=10))
model.add_layer(Softmax())

model.train()