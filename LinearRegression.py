"""
A library for training an agent by the Linear Regression Algorithm.
Included functions :
    (1) preprocessing(x)
    (2) ComputeCost(x,y,theta)
    (3) GradientDescent(x,y,theta,max_iters=100,alpha=0.01)
    (4) Normalization(x,y)
    (5) find_confidence(x_test,y_test,theta)
    (6) predict(x,theta)
    (7) Memorize(theta,name)
    (8) Recall(name)
"""

import numpy as np
import pandas as pd
import pickle as pkl
import os

def preprocessing(x):
    """
    Function name : preprocessing(x)
    ************************************************************************
    Parameters : x [dtype(numpy.array)]
    Here, x is the feature array excluding the additional array of all ones.
    After the preprocessing, this function returns a numpy array with the
    additional array of all ones concatenated automatically.
    ************************************************************************
    """
    c_count = np.size(x,1)
    for i in range(c_count):
        x[:,i] = (x[:,i] - x[:,i].mean())/x[:,i].std()
    x0 = np.array(np.ones(x.shape[0]),ndmin=2)
    x = np.concatenate((x0.T,x),axis=1)
    return x

def ComputeCost(x,y,theta):
    """
    Function name : ComputeCost(x,y,theta)
    ************************************************************************
    Parameters : x [dtype(numpy.array)]
                 y [dtype(numpy.array)]
                 theta [dtype(numpy.array)]
    x : The feature array including the additional column of all ones.
        Use function 'preprocessing()' to normalize the array and add
        additional column of all ones if not added already.
    y : The label array
    theta : The array of the weights
    Function returns the cost value on the basis of the given label and
    weight arrays
    ************************************************************************
    """
    #theta = np.array(theta,ndmin=2)
    #y = y.T
    cost = np.sum((np.dot(x,theta)-y) ** 2) / (2*y.shape[0])
    return cost

def GradientDescent(x,y,theta,max_iters=100,alpha=0.01,verbose=False):
    """
    Function name : GradientDescent(x,y,theta,max_iters=1000,alpha=0.01)
    ************************************************************************
    Parameters : x [dtype(numpy.array)]
                 y [dtype(numpy.array)]
                 theta [dtype(numpy.array)]
    Default Parameters : max_iters [dtype(int)]
                         alpha [dtype(float)]
                         verbose [dtype(bool)]
    x : The feature array including the additional column of all ones.
        Use function 'preprocessing()' to normalize the array and add
        additional column of all ones if not added already.
    y : The label array
    theta : The array of the weights
    max_iters : The maximum numbers of iterations for onr Batch Gradient
                Descent training. Default = 100
    alpha : The Learning Rate. Default = 0.01
    verbose : Gives the information about the steps it is performing if set to
              True. Default value is False.
    Function returns the array of the optimized weights to minimize the cost.
    It also returns the value of the cost history i.e. the cost value after
    each iterations.
    *************************************************************************
    """
    cost_h = []
    m = x.shape[0]
    l = len(theta)
    subtract = np.array([])

    for i in range(max_iters):
        if verbose:
            print("Running "+ str(i+1) + " iteration....")
        if verbose:
            print("Computing the cost...")
        cost = ComputeCost(x,y,theta)
        cost_h.append(cost)
        if verbose:
            print("Determining the weights in " + str(i+1) + " iteration and starting the next iteration......")
        hofx = (x@theta)
        loss = hofx - y

        for j in range(l):
            a = np.dot(x[:,j],loss)
            subtract = np.append(subtract,a)

        theta = theta - (alpha*subtract)
        subtract = np.array([])

    if verbose:
        print("Training complete.")
    return (theta,cost_h)

def Normalization(x,y):
    """
    Function name : Normalization(x,y)
    *************************************************************************
    Parameters : x [dtype(numpy.array)]
                 y [dtype(numpy.array)]
    x : The feature array including the additional column of all ones.
        Use function 'preprocessing()' to normalize the array and add
        additional column of all ones if not added already.
    y : The label array
    Function returns the array of the weights directly by using the Normalized
    Algorithm.
    **************************************************************************
    """
    theta = np.matrix([0]*np.size(x,0))
    x = np.matrix(x)
    y = np.matrix(y)
    mul1 = np.linalg.pinv(np.transpose(x)*x)
    mul2 = np.transpose(x)*y
    theta = np.array(mul1*mul2)
    return theta

def find_confidence(x_test,y_test,theta):
    """
    Function name : find_confidence(x_test,y_test,theta)
    ***************************************************************************
    Parameters : x_test [dtype(numpy.array)]
                 y_test [dtype(numpy.array)]
                 theta [dtype(numpy.array)]
    x_test : The numpy vector of testing features
    y_test : The numpy vector of testing labels
    theta : The numpy weights vector
    Function returns the confidence of the trained agent.
    """
    hofx = np.dot(x_test,theta)
    conf_array = hofx-y_test
    confidence = 1-(conf_array.mean()/conf_array.std())
    return confidence

def predict(x_predict,theta):
    """
    Function name : predict(x_predict,theta)
    ***************************************************************************
    Parameters : x_predict [dtype(numpy.array)]
                 theta [dtype(numpy.array)]
    x_predict : The feature vectors used for predicting the labels
    theta : The trained weight vector.
    Function returns the numpy array of the predicted labels according to the
    values of the trained theta
    ***************************************************************************
    """
    y_predict = np.dot(x_predict,theta)
    return y_predict

def Memorize(theta,name):
    """
    Function name : Memorize(theta)
    ***************************************************************************
    Parameters : theta [dtype(numpy.array)]
                 name [dtype(string)]

    theta : The weights array obtained after the training of the agent.
            The theta is stored in the present working directory.
    name : Specifies the name with which the device remembers its experience.

    Function saves the value of theta to remember it for future use.
    ***************************************************************************
    """
    name = name + ".rut"
    if not os.path.isfile(name):
        with open(name,"wb") as f:
            pkl.dump(theta,f)
        return 1
    else:
        name = input("You have already saved a previous training with the similar name. Please choose a unique name : ")
        Memorize(theta,name)

def Recall(name):
    """
    Function name : Recall(name)
    ***************************************************************************
    Parameters : name [dtype(string)]

    name : The name with which the experience was stored.

    Function loads the values of the weights previously trained and can use it
    for further use.
    ***************************************************************************
    """
    name = name + ".rut"
    try:
        with open(name,"rb") as f:
            theta = pkl.load(f)
        return theta
    except FileNotFoundError:
        print("No such name saved. Error recalling the data.")
        return None
