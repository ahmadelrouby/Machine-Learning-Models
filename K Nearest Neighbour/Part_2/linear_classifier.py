#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:12:49 2018

@author: ahmadrefaat
"""

import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

cifar10_dir = './cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)



print ('Training data shape: ', X_train.shape)
print ('Training labels shape: ', y_train.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()


num_training = 50000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask] 

num_test = 10000
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]


X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

import ls as linear_classifier
classifier = linear_classifier.CIFAR_10_CLASSIFIER()
classifier.train(X_train,y_train)


y_predicted = classifier.predict(X_train)
num_correct = np.sum(y_predicted == y_train)

accuracy = float(num_correct) / num_training
print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_training, accuracy))



"""y_predicted = classifier.predict(X_test)
num_correct = np.sum(y_predicted == y_test)

accuracy = float(num_correct) / num_test
print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


answers_comparison = (y_predicted == y_test)
class_accuracy = []
for i in range(10):
    idx = np.flatnonzero(y_test == i)
    current_correct = np.sum(answers_comparison[idx])
    current_accuracy = current_correct/idx.shape[0]
    class_accuracy.append(current_accuracy)
    print ("Class: ", str(i), " (",classes[i],")" , ", accuracy: " , str(current_accuracy))"""
