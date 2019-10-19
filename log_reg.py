import random
import numpy as np

class log_reg:
    def __init__(self, size):
        self.size = size
        self.weights = np.random.randn(size, 1)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def update(self, feature, label, eta):
        featureM = np.mat(feature)
        labelM = np.mat(label).transpose()
        h = self.sigmoid(featureM * self.weights)
        self.weights = self.weights + eta * featureM.transpose() * (labelM - h)
    def train(self, train_data, epochs, batch_size, eta):
        n = len(train_data)
        for sample in train_data:
            sample[0].append(1)
        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k : k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                feature = [sample[0] for sample in batch]
                label = [sample[1] for sample in batch]
                self.update(feature, label, eta)
            print("Epoch", i, "complete")

if __name__ == '__main__':
    data = [([1, 1, 1], 0), ([0, 0, 0], 1)]
    classifier = log_reg(4)
    classifier.train(data, 1, 2, 0.5)
