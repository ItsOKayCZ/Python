import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if(torch.cuda.is_available()):
    print("Running on GPU")
    device = torch.device("cuda:0")
else:
    print("Running on CPU")
    device = torch.device("cpu")

REBUILD_DATA = False

class CatsVsDogs():
    ImgSize = 50
    Cats = "PetImages/Cat"
    Dogs = "PetImages/Dog"
    Labels = {Cats: 0, Dogs: 1}

    trainingData = []
    catCount = 0
    dogCount = 0

    def getTrainingData(self):
        for label in self.Labels:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.ImgSize, self.ImgSize))

                    self.trainingData.append([np.array(img), np.eye(2)[self.Labels[label]]])

                    if(label == self.Cats):
                        self.catCount += 1
                    elif(label == self.Dogs):
                        self.dogCount += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.trainingData)
        np.save("trainingData.npy", self.trainingData)

        print("Cats:", self.catCount)
        print("Dogs:", self.dogCount)

if(REBUILD_DATA):
    net = CatsVsDogs()
    net.getTrainingData()

print("[#] Loading data")
trainingData = np.load("trainingData.npy", allow_pickle=True)
print("[#] Done loading data")

# import matplotlib.pyplot as plt
#
# print(trainingData[1])
# plt.imshow(trainingData[1][0], cmap="gray")
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if(self._to_linear == None):
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x, dim=1)

def fwdPass(X, y, train=False):
    if(train):
        net.zero_grad()

    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]

    acc = matches.count(True) / len(matches)
    loss = lossFunction(outputs, y)

    if(train):
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    randomStart = np.random.randint(len(testX) - size)
    X, y = testX[randomStart:randomStart + size], testy[randomStart:randomStart + size]
    with torch.no_grad():
        valAcc, valLoss = fwdPass(X.view(-1, 1, 50, 50).to(device), y.to(device))
    return valAcc, valLoss

import time

MODEL_NAME = "model-{0}".format(int(time.time()))

net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
lossFunction = nn.MSELoss()

print("Model: {0}".format(MODEL_NAME))

X = torch.Tensor([i[0] for i in trainingData]).view(-1, 50, 50)
X = X/255.0

y = torch.Tensor([i[1] for i in trainingData])

VAL_PCT = 0.1
valSize = int(len(X) * VAL_PCT)

trainX = X[:-valSize]
trainy = y[:-valSize]

testX = X[-valSize:]
testy = y[-valSize:]

def train():
    BATCH_SIZE = 100
    EPOCHS = 5

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            print("Epoch: {0}/{1}".format(epoch + 1, EPOCHS))

            for i in tqdm(range(0, len(trainX), BATCH_SIZE)):
                batchX = trainX[i:i + BATCH_SIZE].view(-1, 1, 50, 50).to(device)
                batchy = trainy[i:i + BATCH_SIZE].to(device)

                acc, loss = fwdPass(batchX, batchy, train=True)

                if(i % 50 == 0):
                    valAcc, valLoss = test(size=100)
                    f.write("{0},{1},{2},{3},{4},{5}\n".format(MODEL_NAME, round(time.time(), 3), round(float(acc), 2), round(float(loss), 4), round(float(valAcc), 2), round(float(valLoss), 4)))

train()