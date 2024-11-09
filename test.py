from Dense import Dense
from Activations import Tanh,ReLU,Sigmoid
import numpy as np
import matplotlib.pyplot as plt
from network import predict,train
from Losses import MSE,MSE_Derivative


X  = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
y = np.reshape([[0],[1],[1],[0]],(4,1,1))

model = [
    Dense(2,3,initialization='xavier'),
    Sigmoid(),
    Dense(3,1,initialization='xavier'),
    Sigmoid()
]

learning_rate = 0.1 
epochs = 10000

# Training the model
train(model,MSE,MSE_Derivative,X,y,epochs=epochs,learning_rate=learning_rate,verbose=True)

# Testing the model
print(1 if predict(model,[[1],[1]])[0,0] > 0 else 0)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(model, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()