import random
import numpy as np

class log_reg:
    def __init__(self, input_size, output_size):
        self.weights = np.ones((input_size, output_size))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def update(self, feature, label, eta):
        featureM = np.mat(feature)
        labelM = np.mat(label)
        h = self.sigmoid(featureM * self.weights)
        self.weights = self.weights + eta * featureM.transpose() * (labelM - h)
        #print(self.weights)
    def train(self, train_data, epochs, batch_size, eta):
        for sample in train_data:
            np.append(sample[0], 1)
        for _ in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[k : k + batch_size] for k in range(0, len(train_data), batch_size)]
            for batch in batches:
                feature = [sample[0] for sample in batch]
                label = [sample[1] for sample in batch]
                self.update(feature, label, eta)
            #print("Epoch", i, "complete")
        print("train complete")
    def predict(self, test_data):
        for sample in test_data:
            print(np.argmax(self.sigmoid(sample * self.weights)) + 1)
