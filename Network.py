import numpy as np
import _pickle as Cpickle
import os

class network(object):

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

    def reducedim(self, x, alignment = "row"):
        if alignment == "row":
            x = np.reshape(x, (1, np.product(x.shape)))
        elif alignment == "col":
            x = np.reshape(x, (np.product(x.shape), 1))
        else:
            raise ValueException("Parameter 'alignment' has been passed a wrong value")
        return x

    def preprocessing(self, x):
        for i in x:
            i = i/255.0
        return x

    def make_array_output(self, y):
        y = [outputs for outputs in y]
        ret = []
        for outputs in y:
            l = self.size[-1]
            y_out = np.zeros((l,1))
            y_out[outputs] = 1
            ret.append(y_out)
        return ret

    def separate_x_y(self,training_data):
        x, y = training_data
        x = [examples for examples in x]
        x = self.preprocessing(x)
        y = self.make_array_output(y)
        return x, y

    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))

    def make_mini_batches(self,training_data, mini_batch_size):
        self.x, self.y = self.separate_x_y(training_data)
        mini_x = [self.x[k:k+mini_batch_size] for k in range(0,len(training_data),mini_batch_size)]
        mini_y = [self.y[k:k+mini_batch_size] for k in range(0,len(training_data),mini_batch_size)]
        return mini_x, mini_y

    def feedforward(self, x):
        x = self.reducedim(x, alignment="col")
        for w,b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def cost_derivative(self,output,y):
        return (output-y)

    def SGD(self, training_data, mini_batch_size, alpha = 0.01, max_iters = 100, verbose = False):
        l = len(training_data)
        mini_x, mini_y = self.make_mini_batches(training_data,mini_batch_size)
        for iters in range(max_iters):
            if verbose:
                print("Iteration number : {}. Status : Running".format(iters+1))
            self.update_mini_batch(mini_x, mini_y, alpha)
            if verbose:
                print("Iteration number : {}. Status : Complete".format(iters+1))

    def update_mini_batch(self, mini_x, mini_y, alpha):
        delta_b, delta_w = self.backpropagate(mini_x, mini_y)
        for i, j in zip(self.biases, delta_b):
            i = i - (alpha/len(mini_x))*j
        for i, j in zip(self.weights, delta_w):
            i = i - (alpha/len(mini_x))*j

    def backpropagate(self, mini_x, mini_y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        a = mini_x
        a = self.np_to_list(a)
        for xexamples, yexamples, count in zip(mini_x, mini_y, range(len(a))):
            activation = self.reducedim(np.array(xexamples[count]),alignment = "col")
            activations = [activation]
            zs = []
            for i, j in zip(self.weights, self.biases):
                activation = np.dot(i, activation) + j
                zs.append(activation)
                activation = sigmoid(activation)
                activations.append(activation)
            delta = self.cost_derivative(activations[-1], yexamples) * sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in range(2, self.nlayers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta[count]) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, np.array(activations[-l-1]).transpose())
        return (nabla_b, nabla_w)

# Miscellaneous functions:
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

class learn(object):

    def __init__(self, name):
        self.training_name = name
        self.extension = ".rut"

    def memorize(self, weights, biases):
        name = self.training_name + self.extension
        if os.path.isfile(name):
            self.training_name = input("Training of a similar name already exists. Please choose another name : ")
            self.memorise(weights, biases)
        else:
            tuple = (weights, biases)
            with open(name,"wb") as f:
                Cpickle.dump(tuple, f)

    def recall(self):
        name = self.training_name + self.extension
        try:
            with open(name, "rb") as f:
                print("Loading the data.....")
                (weights, biases) = Cpickle.load(f)
                print("Data loaded")
            return (weights,biases)
        except FileNotFoundError:
            print("Error loading data. Training of the specified name does not exist.")
            return None
