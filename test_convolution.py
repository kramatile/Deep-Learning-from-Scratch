from Dense import Dense
from Reshape import Reshape
from MaxPooling import MaxPooling
from AveragePooling import AveragePooling
from Convolution import Convolution2D
from Activations import Tanh,ReLU,Sigmoid
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from network import predict,train
from Losses import BinaryCrossEntropy,BinaryCrossEntropy_Derivative

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test_copy, y_test_copy = x_test.copy(),y_test.copy()
x_test, y_test = preprocess_data(x_test, y_test, 100)


model = [
    Convolution2D((1,28,28), 3, 5),
    Sigmoid(),
    AveragePooling((2,2)),
    Reshape((5, 13, 13), (5 * 13 * 13, 1)),
    Dense(5 * 13 * 13, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

train(
    model,
    BinaryCrossEntropy,
    BinaryCrossEntropy_Derivative,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

for x, y in zip(x_test, y_test):
    output = predict(model, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")