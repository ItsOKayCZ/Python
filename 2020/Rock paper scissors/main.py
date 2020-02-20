from tqdm import tqdm
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import cv2

import matplotlib.pyplot as plt

######################
# DATA
######################
img_size = 50

rock_train = "data/train/rock"
paper_train = "data/train/paper"
scissors_train = "data/train/scissors"

rock_test = "data/test/rock"
paper_test = "data/test/paper"
scissors_test = "data/test/scissors"

train_data = []
test_data = []

labels = {rock_train: 0, paper_train: 1, scissors_train: 2}
test_labels = {rock_test: 0, paper_test: 1, scissors_test: 2}
labels_size = 3

BUILD_DATA = False
if(BUILD_DATA):
    print("Loading train data")
    for label in labels:
        print("Directory: {}".format(label))
        for file in tqdm(os.listdir(label)):
            path = os.path.join(label, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            train_data.append([np.array(img), labels[label]])

    print("Loading test data")
    for label in test_labels:
        print("Directory: {}".format(label))
        for file in tqdm(os.listdir(label)):
            path = os.path.join(label, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            test_data.append([np.array(img), test_labels[label]])

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    np.save("train_data.npy", train_data)
    np.save("test_data.npy", test_data)
    print("Saved data")
else:
    train_data = np.load("train_data.npy", allow_pickle=True)
    test_data = np.load("test_data.npy", allow_pickle=True)


X_train = []
y_train = []

X_test = []
y_test = []

print("Editing train data")
for data in train_data:
    X_train.append(data[0])
    y_train.append(data[1])
X_train = np.array(X_train)
X_train = X_train / 255.0
X_train = X_train.reshape(len(X_train), img_size, img_size, 1)
y_train = np.array(y_train)

print("Editing test data")
for data in test_data:
    X_test.append(data[0])
    y_test.append(data[1])
X_test = np.array(X_test)
X_test = X_test / 255.0
X_test = X_test.reshape(len(X_test), img_size, img_size, 1)
y_test = np.array(y_test)

# plt.imshow(X_train[0].reshape(img_size, img_size))
# plt.show()
######################
# Neural network
######################
EPOCHS = 50
BATCH_SIZE = None
SAVE_MODEL = False

log_dir = "log"
if(SAVE_MODEL):
    import shutil
    try:
        shutil.rmtree("log")
    except Exception:
        pass

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(labels_size, activation="softmax")
    ])



    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback])

    model.evaluate(X_test, y_test, verbose=1)

    model.save("model.h5")
else:
    model = tf.keras.models.load_model("model.h5")

######################
# Validation
######################
# import sys
# sys.exit()
print("Evaluating")
path = os.path.join("data", "validation")

size = 0
succ = 0
for file in tqdm(os.listdir(path)):

    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))

    X = np.array(img)
    X = X / 255.0
    X = X.reshape(1, img_size, img_size, 1)
    prediction = np.argmax(model.predict(X)[0])

    if("rock" in file):
        correct = 0
    elif("paper" in file):
        correct = 1
    elif("scissors" in file):
        correct = 2
    size += 1

    if(prediction == correct):
        succ += 1

    if("(" in file):
        print("File: {}".format(file))
        print("Guessed: {}".format(prediction))
        if(prediction == correct):
            print("Correct")
        else:
            print("Incorrect")

print("Success Percentage: {}".format((succ / size) * 100))