# -*- coding: utf-8 -*-
"""CSCE_790_NN_HW_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JJ139sZeMko5wJyygQv09hF7xdLJXrme
"""

# Part B
# Q1

from google.colab import drive
drive.mount('/content/drive')

save_path = '/content/drive/MyDrive/Colab Notebooks/training_error_plot.png'

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
 [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]])

Y = np.array([[1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

eta = 0.1 # learning rate
epochs = 100

error_TRs = []

np.random.seed(2024)

# initializing with some random numbers for all then we can avoid the identical gradients
W = np.random.randn(2, 2)
b = np.random.randn(2, 1)

for epoch in range(epochs):
    Z = np.dot(W, X) + b
    Y_pred = sigmoid(Z)
    error = Y_pred - Y
    error_TR = np.mean(np.abs(error))
    error_TRs.append(error_TR)
# backpropagation
    dZ = error * sigmoid_deriv(Z) # gradient loss w.r.t Z
    dW = np.dot(dZ, X.T) / X.shape[1]
    db = np.sum(dZ, axis=1, keepdims=True) / X.shape[1] # divide by num cols, num features

    W -= eta * dW
    b -= eta * db

plt.figure()
plt.savefig(save_path, format='png')
plt.plot(range(epochs), error_TRs)
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.title('Training errors by increasing the number of epoch')
plt.show()

plt.figure()  # Create a new figure
plt.plot(range(epochs), error_TRs)  # Plot the data
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Training Error')  # Add y-axis label
plt.title('Change in the training error based on increased number of epochs')  # Add title

# Step 4: Save the plot as EPS in Google Drive
plt.savefig(save_path, format='eps')

# Optional: Display the plot to verify
plt.show()

# Close the plot to free memory
plt.close()

# decision boundary plot function is based on the below
# https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot

#print(xx.shape)
#print(yy.shape)

#grid_points = np.c_[xx.flatten(), yy.flatten()]  # Shape: (n_points, 2)

#print(grid_points.shape)



def db_plot(W, b, X, Y, epoch):
    h = 0.02  # step size for meshgrid, xx.shape; yy.shape both are (155,155)
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.flatten(), yy.flatten()]  # shape: (num_points=24025, 2), flattening for visualisation

    Z = np.dot(grid_points, W.T) + b.T
    Y_pred = sigmoid(Z)

    class_pred = (Y_pred > 0.5).astype(int) # boolean into integer
    class_pred = class_pred[:, 0] + 2 * class_pred[:, 1]
    # class is 0 or 1 and for binary classification 4 possibilites, so coding that way accordingly

    plt.contourf(xx, yy, class_pred.reshape(xx.shape), alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[0, :], X[1, :], c=np.argmax(Y, axis=0), edgecolors='k', cmap=plt.cm.Paired)
    plt.title(f'Decision Boundary after {epoch} Epochs')
    plt.show()

db_plot(W, b, X, Y, epoch=3)
db_plot(W, b, X, Y, epoch=10)
db_plot(W, b, X, Y, epoch=100)

# Part B
# Q2

# [-1,1] so hidden layer is tanh, Y seems continuous so used linear in the output layer

def tanh(Z):
    return np.tanh(Z)

def tanh_deriv(Z):
    return 1 - np.tanh(Z)**2

def linear(x):
    return x

def linear_deriv(x):
    return np.ones_like(x)

X = np.array([[-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
               0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

Y = np.array([[-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134,
               -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396,
               0.345, 0.182, -0.031, -0.219, -0.321]])

np.random.seed(2024)
# hidden layer
W1 = np.random.randn(10, 1)
b1 = np.random.randn(10, 1)
# output layer
W2 = np.random.randn(1, 10)
b2 = np.random.randn(1, 1)

eta = 0.01
epochs = 1000

error_TRs = []

for epoch in range(epochs):
    # Forward pass
    Z1 = np.dot(W1, X) + b1
    pred1 = sigmoid(Z1)

    Z2 = np.dot(W2, pred1) + b2
    pred_y = linear(Z2)

    error = pred_y - Y
    error_TR = np.mean(np.square(error))  # MSE
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Training Error: {error_TR:.6f}")

    error_TRs.append(error_TR)  # append training error, once in 1 epoch

    # backpropagation
    dZ2 = error * linear_deriv(Z2)
    dW2 = np.dot(dZ2, pred1.T) / X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

    dpred1 = np.dot(W2.T, dZ2)
    dZ1 = dpred1 * sigmoid_deriv(Z1)
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

    # update the gradient descent
    W2 -= eta * dW2
    b2 -= eta * db2
    W1 -= eta * dW1
    b1 -= eta * db1

plt.figure()  # Create a new figure
plt.plot(range(epochs), error_TRs)  # Plot the data
plt.xlabel('Epoch')  # Add x-axis label
plt.ylabel('Training Error')  # Add y-axis label
plt.title('Change in the training error based on increased number of epochs')  # Add title

# Step 4: Save the plot as EPS in Google Drive
plt.savefig(save_path, format='eps')

# Optional: Display the plot to verify
plt.show()

# Close the plot to free memory
plt.close()

plt.figure()
plt.plot(range(epochs), error_TRs)
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.title('Change in the training error based on increased the number of epoch')
plt.show()

len(error_TRs)

epochs

print(error_TRs)

# forward pass & final output
Z1 = np.dot(W1, X) + b1
pred1 = sigmoid(Z1)
Z2 = np.dot(W2, pred1) + b2
Y_pred = linear(Z2)

plt.figure()
plt.plot(X.flatten(), Y.flatten(), 'o-', label='Actual Function')
plt.plot(X.flatten(), Y_pred.flatten(), '.-', label='NN Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Actual vs. NN Approximation')
plt.show()

def nn_approx(W1, b1, W2, b2, epoch):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    Y_pred = linear(Z2)

    plt.figure()
    plt.plot(X.flatten(), Y.flatten(), 'o-', label='Actual Function')
    plt.plot(X.flatten(), Y_pred.flatten(), 'x-', label=f'NN Approximation at {epoch} Epochs')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'NN Approximation after {epoch} Epochs')
    plt.show()

nn_approx(W1, b1, W2, b2, epoch=10)
nn_approx(W1, b1, W2, b2, epoch=100)
nn_approx(W1, b1, W2, b2, epoch=200)
nn_approx(W1, b1, W2, b2, epoch=400)
nn_approx(W1, b1, W2, b2, epoch=1000)