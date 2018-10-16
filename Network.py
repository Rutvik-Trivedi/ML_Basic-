import numpy as np
import _pickle as Cpickle
import random
import os

# Here data will be a tuple of three-D (two-D for one training example and a third dimension for number of examples) array of 'x' and a one-D array of 'y' (one output for
# each example)

class network(object):

    entertain = ["Wow! Learning is fun!", "What a busy day!", "Do not disturb. Using my brain right now.",
                 "This one was tough", "Ohh! I think that's the answer", "Yeah, whatever...",
                 "What the heck! Didn't expect that answer!", "Yeah, got that right", "Hey, did you know this?",
                 "That was a real fun fact", "Maybe I should have some rest. Shouldn't I?", "I think my brain works really fast",
                 "I hope I pass in the tests", "Train hard. Tests are on their way", "Will the tests be open book?"]

    def __init__(self, *size):
        self.nlayers = len(size)
        self.size = size
        self.nhidden = self.nlayers-2
        self.weights = [np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
        self.biases = [np.random.randn(x,1) for x in size[1:]]
        self.x = None
        self.y = None

    def np_to_list(self, x):
        a = [i for i in x]
        return a

    def reducedim(self, x, alignment = "col"): # converting to (a,1) shaped array
        x = np.ravel(x)
        x = np.array(x, ndmin=2)
        if alignment == "row":
            return x
        elif alignment == "col":
            x = x.transpose()
            return x
        else:
            raise ValueException("Parameter 'alignment' has been passed a wrong value")

    def preprocessing(self, x):
        a = self.np_to_list(x)
        for count in range(len(a)):
            a[count] = self.reducedim(a[count], alignment = "col")
            a[count] = a[count]/255.0
        return a

    def make_array_output(self, y):
        ret = []
        l = self.size[-1]
        for outputs in y:
            y_out = np.zeros((l,1))
            y_out[outputs] = 1
            ret.append(y_out)
        return ret

    def separate_x_y(self,training_data):
        x, y = training_data
        x = self.preprocessing(x)
        y = self.make_array_output(y)
        return x, y

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return sigmoid(z)*(1-sigmoid(z))

    def make_mini_batches(self,training_data, mini_batch_size):
        self.x, self.y = self.separate_x_y(training_data)
        l = len(self.x)
        mini_x = [self.x[k:k+mini_batch_size] for k in range(0,l,mini_batch_size)]
        mini_y = [self.y[k:k+mini_batch_size] for k in range(0,l,mini_batch_size)]
        return mini_x, mini_y

    def feedforward(self, x, reduce_dim = True): #Calculates output for one training example at a time
        if reduce_dim:
            x = self.reducedim(x, alignment="col")
        for w,b in zip(self.weights, self.biases):
            x = np.dot(w, x) + b
        x = self.find_result(x)
        return x

    def find_result(self, y):
        y = y.ravel().tolist()
        maximum = max(y)
        return (y.index(maximum)+1) #Generally, we count the neurons from 1

    def cost_derivative(self,output,y):  #For only one training example
        return (output-y)

    def SGD(self, training_data, mini_batch_size, alpha = 0.01, max_iters = 100, verbose = None): #For all the training examples divided into batches of size mini_batch_size
        mini_x, mini_y = self.make_mini_batches(training_data,mini_batch_size)
        for iters in range(max_iters):
            for xnum, ynum in zip(range(len(mini_x)), range(len(mini_y))):
                self.update_mini_batch(mini_x[xnum], mini_y[ynum], alpha, mini_batch_size)  # Updates one mini batch at a time. This goes for max_iters number of times for all
            if verbose == "normal":                                                         # the mini batches.
                print("Iteration number : {}  Status : Complete".format(iters+1))
            elif verbose == "entertain":
                print(random.choice(network.entertain))
        if verbose == "normal":
            print("Training Complete.")
        elif verbose == "entertain":
            print("Completed finally. Now I got to know how Albert Einstein must have felt")

    def update_mini_batch(self, mini_x, mini_y, alpha, mini_batch_size): #For one Batch of size mini_batch_size
        delta_b, delta_w = self.backpropagate(mini_x, mini_y)
        for i, j in zip(range(len(self.biases)), range(len(delta_b))):
            self.biases[i] = self.biases[i] - (alpha/mini_batch_size)*delta_b[j]
            self.weights[i] = self.weights[i] - (alpha/mini_batch_size)*delta_w[j]

    def backpropagate(self, mini_x, mini_y):    # backpropagate works on the mini batch for one training example at a time
        l = len(mini_x)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for number in range(len(mini_x)):
            activation = mini_x[number]
            activations = [activation]
            zs = []
            for i, j in zip(self.weights, self.biases):
                activation = np.dot(i, activation) + j
                zs.append(activation)
                activation = self.sigmoid(activation)
                activations.append(activation)
            delta = self.cost_derivative(activations[-1], mini_y[number]) * self.sigmoid_prime(zs[-1])
            nabla_w[-1] = delta * np.array(activations[-2]).transpose()
            nabla_b[-1] = delta
            for l in range(2, self.nlayers):
                z = zs[-l]
                sp = self.sigmoid_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_w[-l] = delta * activations[-l-1].transpose()
                nabla_b[-l] = delta
        return (nabla_b, nabla_w)

# Miscellaneous functions:
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

def feedforward(x, weights, biases):
    x = network.reducedim(network, x = x, alignment="col")
    for w,b in zip(weights, biases):
        x = np.dot(w, x) + b
    x = find_result(x)
    return x

def find_result(y):
    y = y.ravel().tolist()
    maximum = max(y)
    return (y.index(maximum)+1)


class learn(object):

    def __init__(self, name):
        self.training_name = name

    def memorize(self, size, weights, biases):
        name = self.training_name + ".rut"
        if os.path.isfile(name):
            self.training_name = input("Training of a similar name already exists. Please choose another name : ")
            self.memorize(size, weights, biases)
        else:
            tuple = (size, weights, biases)
            with open(name,"wb") as f:
                Cpickle.dump(tuple, f)

    def recall(self):
        name = self.training_name + ".rut"
        try:
            with open(name, "rb") as f:
                print("Loading the data.....")
                (size, weights, biases) = Cpickle.load(f)
                print("Data loaded")
            return (size, weights, biases)
        except FileNotFoundError:
            print("Error loading data. Training of the specified name does not exist.")
            return None
