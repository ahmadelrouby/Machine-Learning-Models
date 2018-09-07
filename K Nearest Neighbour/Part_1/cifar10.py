#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 18:13:44 2018

@author: ahmadrefaat
"""

import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import NearestNeighbor as NN

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



################# TRYING TO GET ONLY 10K TRAINING SET
################# WITH 1K FROM EACH CLASS 
################# RANDOMLY CHOSEN
#####################################################################################
#####################################################################################

X_train = np.reshape(X_train, (X_train.shape[0], -1))
training_samples_per_class = 999
full_Xs = np.array([], dtype=int)

for i in range(num_classes):
    idx = np.flatnonzero(y_train == i)  
    idx = np.random.choice(idx, training_samples_per_class, replace = False)
    full_Xs = np.append(full_Xs,idx)
    

np.random.shuffle(full_Xs)
X_train = X_train[full_Xs,:]
y_train = y_train[full_Xs]


num_test = 10000
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]


X_test = np.reshape(X_test, (X_test.shape[0], -1))



#####################################################################################
##    Predicting Test data set
#####################################################################################
#####################################################################################
classifier = NN.NearestNeighbor()
classifier.train(X_train, y_train)

#(5)KNN Classifier 
y_test_predicted_5NN = classifier.predictKnn(X=X_test,l='L1',k=7)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_predicted_5NN == y_test)
accuracy = float(num_correct) / num_test
print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


answers_comparison = (y_test_predicted_5NN == y_test)
class_accuracy = []
for i in range(10):
    idx = np.flatnonzero(y_test == i)
    current_correct = np.sum(answers_comparison[idx])
    current_accuracy = current_correct/idx.shape[0]
    class_accuracy.append(current_accuracy)
    print ("Class: ", str(i), " (",classes[i],")" , ", accuracy: " , str(current_accuracy))





##########################
## At this point, X_Train and Y_train 
## have equal RANDOM representation of each class
## Training set is now of size 10K with 1K for each class
#####################################################################################
#####################################################################################

classifier = NN.NearestNeighbor()
all_KS = [1,2,3,5,7,9,12,15]
all_Lengths = ['L1','L2']
splitted_data_X = np.split(X_train,3)
splitted_data_Y = np.split(y_train,3)


max_accuracy = 0
max_k = 0
max_length = 'l1'

for length in all_Lengths:
    for current_K in all_KS:
        total_accuracy = []
        for i in range(3):
            validation_X = splitted_data_X[i]
            validation_Y = splitted_data_Y[i]
            if(i == 0):
                splitted_data_indices = np.array([1,2])
            elif (i==1):
                splitted_data_indices = np.array([0,2])
            elif(i==2):
                splitted_data_indices = np.array([0,1])
            #splitted_data_indices = np.concatenate([j for j in range(3) if j != i])
            
            part_1_X = splitted_data_X[int(splitted_data_indices[0])]
            part_2_X = splitted_data_X[int(splitted_data_indices[1])]
            training_X = np.concatenate((part_1_X,part_2_X))
            
            
            part_1_Y = splitted_data_Y[int(splitted_data_indices[0])]
            part_2_Y = splitted_data_Y[int(splitted_data_indices[1])]
            training_Y = np.concatenate((part_1_Y,part_2_Y))
            
            classifier.train(training_X,training_Y)
            predicted = classifier.predictKnn(X=validation_X,l=length,k=current_K,fold=i)
            num_correct = np.sum(predicted == validation_Y)
            accuracy = float(num_correct) / float(validation_Y.shape[0])
            total_accuracy.append(accuracy)
        with open("data.txt", "a") as myfile:
            myfile.write('\n\n\nAverage Accuracy for L: ' + length+ ',   K: '+ str(current_K)+ ', is ' + str(np.mean(total_accuracy)*100) + '%')
            myfile.write('\nStandard Deviation for L: ' + length+ ',  K: '+ str(current_K)+ ', is ' + str(np.std(total_accuracy)))
        
##########################
## End testing and tuning
#####################################################################################
#####################################################################################


