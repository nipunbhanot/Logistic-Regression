# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:24:51 2019

@author: nipun
"""

import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def generate_y(x):
    y = np.zeros(1000)
    for i in range(1000):    
        if x[i] >= 0.5:
            y[i] = 1 if random.random() <= 0.85 else 0
        else:
            y[i] = 0 if random.random() <= 0.85 else 1
    return y

            
            
def generate_y_bayes(x):
    
    y = np.zeros(1000)
    for i in range(1000):    
        if x[i] >= 0.5: ## It will just be this condition as we are picking the maximum probability out of p and 1-p, which is p
            y[i] = 1
        else:           ## It will just be this condition as we are picking the maximum probability out of q and 1-q, which is 1-q
            y[i] = 0
    return y


if __name__ == '__main__':
    
    ## Generate x from uniform distribution
    x_train = np.random.uniform(0,1,1000)

    ## Generate train labels
    y_train = np.zeros(1000)
    y_train = generate_y(x_train)

    ## Binary Classifier using Logistic Regression
    x_train = x_train.reshape(-1,1)
    clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(x_train,y_train)

    ## Generate test x from uniform distribution
    x_test = np.random.uniform(0,1,1000)
    
    ## Generate test labels
    y_test = generate_y(x_test)
    
    ## Predict
    x_test = x_test.reshape(-1,1)
    y_pred = clf.predict(x_test)
    
    accuracy = clf.score(x_test, y_test)
    print('The accuracy of the classifier is: ', accuracy)
    
    
    
    ## Bayes Optimal Classifier
    
    y_predict_bayes = np.zeros(1000)
    y_predict_bayes = generate_y_bayes(x_test)

    correct_pred = 0
    for j in range(len(y_predict_bayes)):
        if y_predict_bayes[j] == y_test[j]:
            correct_pred = correct_pred + 1
    
    accuracy_bayes = correct_pred/len(y_predict_bayes)
    

    
    print('The accuracy of the Bayes Optimal Classifier is: ', accuracy_bayes)
    

    
    