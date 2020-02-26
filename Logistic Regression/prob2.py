# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:46:18 2019

@author: nipun
"""

import numpy as np
from matplotlib import pyplot as plt


## Computation of Loss

def compute_loss(data, labels, B, B_0):
    
    logloss = 0
    
    for i in range(len(labels)):
        dot_product = np.dot(B, data[i, :].T)
        product = -labels[i]*(B_0 + dot_product)
        logloss += (1/len(labels))*(np.log(1 + np.exp(product)))
        
    return logloss


## Computation of Gradients of B0 and B1
    
def compute_gradients(data, labels, B, B_0):
    
    dB_0 = 0
    dB = np.zeros((1, 784))
    
    for i in range(len(labels)):
        dB_0 += (-1/len(labels))*((np.exp(-labels[i]*(B_0 + np.dot(B,data[i, :].T))))/(1 + (np.exp(-labels[i]*(B_0 + np.dot(B,data[i, :].T))))))*labels[i]
        dB += (-1/len(labels))*((np.exp(-labels[i]*(B_0 + np.dot(B,data[i, :].T))))/(1 + (np.exp(-labels[i]*(B_0 + np.dot(B,data[i, :].T))))))*labels[i]*data[i, :]
   
    return dB, dB_0


## Prediction of Labels

def predict_y(x_test, y_test, B, B_0):
    
    y_predict = np.zeros(len(y_test))
    
    for i in range(len(y_test)):
        prob_class1 = 1/(1 + np.exp(-(B_0 + np.dot(B, x_test[i, :].T))))

        if(prob_class1 >= 0.5):
            y_predict[i] = 1
        else:
            y_predict[i] = -1
     
    return y_predict
        




if __name__ == '__main__':
   
    x = np.load('C:\Penn\Courses\ESE 542\HW5\data.npy')
    y = np.load('C:\Penn\Courses\ESE 542\HW5\label.npy')
    
    x_normalized = x/255.0
    
    for i in range(len(y)):
        if (y[i] == 0):
            y[i] = 1
        else:
            y[i] = -1

    ## Split the data to train and test
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(x_normalized)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    x_train = x_normalized[0:11824]
    x_test = x_normalized[11824:14780]
    
    y_train = y[0:11824]
    y_test = y[11824:14780]

    ## Initialize Beta_0 and Vector Beta_0
    B = np.random.randn(1, x.shape[1])
    B_0 = np.random.randn(1)
    
    lr = 0.05
    
    loss = np.zeros(50)
    loss_testdata = np.zeros(50)
    accuracy_test = np.zeros(50)

 
    for i in range(50):
        
        ## Compute Loss
        
        loss[i] = compute_loss(x_train, y_train, B, B_0)
        loss_testdata[i] = compute_loss(x_test, y_test, B, B_0)
        
        ## Print loss for the last iteration
        if i==49:
            print('Loss after the 50th epoch on train data: ', loss[i])
            print('Loss after the 50th epoch on test data: ', loss_testdata[i])
            
        ## Compute Gradients
        dB, dB_0 = compute_gradients(x_train, y_train, B, B_0)

        ## Update Parameters
        B = B - lr*dB
        B_0 = B_0 - lr*dB_0
        
        ## Predict on Test Data
        y_predict = predict_y(x_test, y_test, B, B_0)
    
        ## Compute Accuracy
        correct_pred = 0
        for j in range(len(y_test)):
            if(y_test[j] == y_predict[j]):
                correct_pred = correct_pred + 1
            
        accuracy_test[i] = correct_pred/len(y_predict)
        
        ## Print Accuracy after the last iteration
        if i==49:
            print('Accuracy after the 50th epoch on test data: ', accuracy_test[i])
            
            
        
    fig1 = plt.figure(1)
    plt.plot(loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss with respect to iteration for train set (1 to 50)') 
    fig1.show()

    fig2 = plt.figure(2)
    plt.plot(loss_testdata)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss with respect to iteration for test set (1 to 50)')
    fig2.show()
        
    fig3 = plt.figure(3)
    plt.plot(accuracy_test)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy with respect to iteration for test set (1 to 50)')
    fig3.show()

