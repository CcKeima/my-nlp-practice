# -*- coding: utf-8 -*-
import random
import numpy as np

class bp:
    def __init__(self, size):
        self.n_layer = len(size)
        self.size = size
        self.biases = [np.random.randn(x, 1) for x in size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)
    
    def cost_derivative(self, x, y):
        return y - x 

    def calculate(self, x):
        for b, w in zip(self.biases, self.weights):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.calculate(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_result)

    def bp(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        print(delta)
        nabla_w[-1] = np.dot(delta, np.array(activations[-2]).transpose())
        for l in range(2, self.n_layer):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.array(activations[-l - 1]).transpose())
        return (nabla_b, nabla_w)

    def update(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nable_b, delta_nable_w = self.bp(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nable_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nable_w)]
            self.biases = [b - (eta / (len(batch))) * nb for b, nb in zip(self.biases, nabla_b)]
            self.weights = [w - (eta / (len(batch))) * nw for w, nw in zip(self.weights, nabla_w)]
            
    def train_SGD(self, train_data, epochs, batch_size, eta, test_data = None):
        n = len(train_data)
        if test_data:
            n_test = len(test_data)
        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update(batch, eta)
            if test_data:
                print(self.evaluate(test_data) / n_test, end = ' ')
            print("Epoch ", i, "complete")