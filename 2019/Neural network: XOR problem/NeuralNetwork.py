import numpy as np

import numpy as np

class NeuralNetwork:

  def __init__(this, iNodes, hNodes, oNodes):
    this.inputNodes = iNodes
    this.hiddenNodes = hNodes
    this.outputNodes = oNodes

    this.hiddenWeights = np.random.uniform(low=-1, high=1, size=(this.inputNodes, this.hiddenNodes))
    this.hiddenBias = np.random.uniform(size=(1, this.hiddenNodes))

    this.outputWeights = np.random.uniform(low=-1, high=1, size=(this.hiddenNodes, this.outputNodes))
    this.outputBias = np.random.uniform(size=(1, this.outputNodes))

    this.lr = 0.1

    print("Hidden weights:")
    print(this.hiddenWeights)
    print("Hidden bias:")
    print(this.hiddenBias)
    print("Output weights:")
    print(this.outputWeights)
    print("Output bias:")
    print(this.outputBias)
    print("--------------------------")

  def feedforward(this, inputs):

    inputs = np.asarray(inputs)

    # Calculating the values of the hidden layer
    this.hiddenLayer = np.dot(inputs, this.hiddenWeights)
    this.hiddenLayer += this.hiddenBias
    this.hiddenLayer = this.sigmoid(this.hiddenLayer)

    # Calculating the values of the output layer
    this.outputLayer = np.dot(this.hiddenLayer, this.outputWeights)
    this.outputLayer += this.outputBias
    this.outputLayer = this.sigmoid(this.outputLayer)

    return this.outputLayer

  def train(this, inputs, correct):
    inputs = np.asarray(inputs)
    correct = np.asarray(correct)

    this.feedforward(inputs)

    outputError = correct - this.outputLayer
    dPredict = outputError * this.dSigmoid(this.outputLayer)

    hiddenError = dPredict.dot(this.outputWeights.T)
    dHidden = hiddenError * this.dSigmoid(this.hiddenLayer)


    this.outputWeights += this.hiddenLayer.T.dot(dPredict) * this.lr
    this.outputBias += np.sum(dPredict) * this.lr

    this.hiddenWeights += inputs.T.dot(dHidden) * this.lr
    this.hiddenBias += np.sum(dHidden) * this.lr

  def dSigmoid(this, x):
    return x * (1 - x)

  def sigmoid(this, x):
    return 1 / (1 + np.exp(-x))
