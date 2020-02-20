import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import cv2

#################### Hyper parameters ####################
EPOCHS = 35

log_name = "log"

SAVED = False
####################


mnist = tf.keras.datasets.mnist
model = None


if(not SAVED):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    print(y_train)
    os.exit()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    log_dir = os.getcwd() + "\\" + log_name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(x_train, y_train,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])

    model.evaluate(x_test, y_test, verbose=1)

    model.save("model.h5")
else:
    model = tf.keras.models.load_model("model.h5")

correct = 0
for i in range(0, 10):
    name = "number" + str(i) + ".png"
    img = cv2.imread(name, 0)

    X = img / 255.0
    X = tf.Variable([X], tf.float32)
    X = tf.reshape(X, [1, 28, 28, 1])

    prediction = np.argmax(model.predict(X)[0])
    if(prediction == i):
        correct += 1
    else:
        print("Guessed: {}".format(prediction))
        print("Correct: {}".format(i))
        print()


print("Percentage: {}".format(correct/10))
