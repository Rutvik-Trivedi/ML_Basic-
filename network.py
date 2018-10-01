import numpy as np
import pandas as pd
import random as random

class Network(object):

    def __init__(self, *size):
        self.nlayers = len(size)
        self.size = size
        self.bias = [np.random.randn(y,1) for y in size[1:]] #Here the biases starts from the hidden layers and not from the input layers as we will omit
        # biasing the input layer. So it starts from 1 to the end
        self.weight = [np.random.randn(y,x)
                       for x, y in zip(size[:-1], size[1:])]
        # Here the weight is a list of numpy arrays. So, Network.weight[n] will give the array of weights of layer n+1 and n+2.

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))  # The normal Sigmoid function

    def feedforward(self,a): # To find the output of the given input layer 'a'
        """Return the output of the network if "a" is input"""
        for b, w in zip(self.bias, self.weight):
            a = sigmoid(np.dot(w,a)+b) #Finds the sigmoid of the resulting array
        return a

    def SGD(self, training_data, iters, mini_batch_size, alpha,test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        iteration, and partial progress printed out.  This is useful
        for tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data) #If applicable, stores the length of the test data
        n = len(training_data)  #Store the length of the training data to help divide into batches
        for j in range(iters):
            random.shuffle(training_data)   #Shuffles the data randomly
            mini_batches = [
                training_data[k:k+mini_batch_size]  #Creates a list of mini batches of size given by the user
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha) # Carries out the stochastic gradient descent
            if test_data:   #Prints the accuracy if the test data is provided
                print("Iteration {0}: {1} / {2}").format(
                    j, self.evaluate(test_data), n_test)
            else:   #Prints the progress of the training
                print("Iteration {0} complete").format(j)

    def update_mini_batch(self, mini_batch, alpha): #The main function which updates the weights and biases of the given mini batches
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(alpha/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(alpha/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):     # Function implementing the backpropagation algorithm
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

# Miscellaneous functions

def sigmoid(z): #The normal sigmoid function
    return 1.0/(1.0 + (np.exp(-z)))

def sigmoid_prime(z): #Derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))
