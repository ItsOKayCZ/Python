import numpy as np

from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork(2, 2, 1)

inputs = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
targets = np.array([[0], [1], [1], [0]])

for _ in range(100000):
  nn.train(inputs, targets)

print(nn.feedforward(np.array([[0, 0]])))
print(nn.feedforward(np.array([[1, 1]])))
print(nn.feedforward(np.array([[0, 1]])))
print(nn.feedforward(np.array([[1, 0]])))