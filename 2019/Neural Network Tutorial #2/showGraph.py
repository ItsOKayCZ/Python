import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model = "model-1571255818"

def createAccLossGraph(modelName):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    valAccs = []
    valLosses = []

    for c in contents:
        if(modelName in c):

            name, timestamp, acc, loss, valAcc, valLoss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            valAccs.append(float(valAcc))
            valLosses.append(float(valLoss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, valAccs, label="valAcc")
    ax1.legend(loc=2)

    ax2.plot(times, losses, label="loss")
    ax2.plot(times, valLosses, label="valLoss")
    ax2.legend(loc=2)

    plt.show()

createAccLossGraph(model)