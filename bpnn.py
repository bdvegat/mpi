import numpy as np
import pandas as pd
from timeit import default_timer as timer
from mpi4py import MPI
import math as m

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y,size, rank, comm, learning_rate=0.2, epochs=100):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        
        for k in range(epochs):
            for i in range(len(X)):
                i = np.random.randint(X.shape[0])
                a = [X[i]]

                for l in range(len(self.weights)):
                        dot_value = np.dot(a[l], self.weights[l])
                        activation = self.activation(dot_value)
                        a.append(activation)
                # output layer
                error = y[i] - a[-1]
                deltas = [error * self.activation_prime(a[-1])]

                # we need to begin at the second to last layer 
                # (a layer before the output layer)
                for l in range(len(a) - 2, 0, -1): 
                    deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

                # reverse
                # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
                deltas.reverse()

                # backpropagation
                # 1. Multiply its output delta and input activation 
                #    to get the gradient of the weight.
                # 2. Subtract a ratio (percentage) of the gradient from the weight.
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta)

            data = self.weights
            # for i in data:
            #     print(i)
            #     print (i+i)
            data = comm.gather(data, root=0)
            sendbuf = None
            if rank == 0:
                for i in range(1, size):
                    for j in range(0,len(data[i])):
                        self.weights[j] = self.weights[j] + data[i][j]
                        # print(data[i][j],"\n")
                        # print(data[i][j] + data[i][j])
                        # print()
                sendbuf = self.weights
            sendbuf = comm.scatter(sendbuf, root=0)
                # if k % 10000 == 0: print ('epochs:', k)
            if rank!=0:
                for i in sendbuf:
                    self.weigths = i
                    
    def predict(self, x): 
        a = np.concatenate((np.ones(1), np.array(x)), axis=0).T     
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':
    # mpiexec -n 4 python3 bpnn.py
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    nn = NeuralNetwork([4,2,3,2,1])
    df = pd.read_csv('data.txt', sep=",", header=None)
    length =  m.floor(len(df)/size)
    X = df[rank*length:rank*length+length][[0,1,2,3]] 
    X = np.array(X)
    y = df[:][[4]]
    y = np.array(y)
    start = timer()
    nn.fit(X, y, size, rank, comm)
    end = timer()
    if rank==0:
        print(end - start)
    # for e in X:
    #     print(e,nn.predict(e))


# ...
