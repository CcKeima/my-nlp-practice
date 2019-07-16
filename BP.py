# -*- coding: utf-8 -*-
import random
import numpy as np

class bp:
    def __init__(self, size):
        self.layer_num = len(size)
        self.size = size
        self.biases = [np.random.randn(x, 1) for x in size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1))
    def bp(self, x, y):


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
                print(i)
            else:
                print("Epoch ", i, "complete")
    