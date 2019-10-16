#!/usr/bin/python
import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 30

# print("[#] Starting training")
# for epoch in range(epochs):
#     for data in trainset:
#         X, y = data
#         net.zero_grad()
#
#         output = net(X.view(-1, 28*28))
#
#         loss = F.nll_loss(output, y)
#         loss.backward()
#         optimizer.step()
#
#     print("Epoch: {}".format(epoch + 1))
#     print("Loss: {}".format(loss))
#
# print("[#] Done training")
#
# correct = 0
# total = 0
#
# print("[#] Calculating accuracy")
# with torch.no_grad():
#     for data in trainset:
#         X, y = data
#         output = net(X.view(-1, 28*28))
#
#         for index, i in enumerate(output):
#             if torch.argmax(i) == y[index]:
#                 correct += 1
#             total += 1
# print("Accuracy: ", round(correct/total, 3))
#

# torch.save(net.state_dict(), "savedModel")
net.load_state_dict(torch.load("savedModel"))


from skimage import color
from skimage import io
import numpy as np

imageName = input("Enter image name: ")
image = np.array(io.imread(imageName), dtype=np.float32)
image = color.rgb2gray(image)
image = 255 - image
image = torch.from_numpy(image)

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
# # print(train[0][0].view(28, 28).shape)
# plt.imshow(train[0][0].view(28, 28))
# plt.show()

with torch.no_grad():
    output = net(image.view(1, 28 * 28))
    print(output)
    print(torch.argmax(output))