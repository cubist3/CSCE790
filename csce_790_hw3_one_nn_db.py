# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def cross_entropy_loss(Y_pred, Y):
    return -np.mean(np.sum(Y * np.log(Y_pred + 1e-9), axis=0))

X = np.array([[0.1, 0.7, 0.8, 0.8, 1.0, 0.3, 0.0, -0.3, -0.5, -1.5],
              [1.2, 1.8, 1.6, 0.6, 0.8, 0.5, 0.2, 0.8, -1.5, -1.3]])
 # one-hot encoding for X - 4 classes
Y = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

# normalisation input
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

np.random.seed(2024)
W = np.random.randn(4, 2) * np.sqrt(1 / 2)
b = np.zeros((4, 1))

eta = 0.1
epochs = 100
error_history = []

for epoch in range(epochs):
    Z = np.dot(W, X) + b
    Y_pred = softmax(Z)

    loss = cross_entropy_loss(Y_pred, Y)
    error_history.append(loss)

    error = Y_pred - Y
    dW = np.dot(error, X.T) / X.shape[1]
    db = np.sum(error, axis=1, keepdims=True) / X.shape[1]

    W -= eta * dW
    b -= eta * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

plt.plot(range(epochs), error_history)
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Error vs Epochs')
plt.show()

def db_plot(W, b, X, Y, epoch):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = np.dot(W, grid) + b
    Y_pred = softmax(Z)
    class_pred = np.argmax(Y_pred, axis=0).reshape(xx.shape)

    plt.contourf(xx, yy, class_pred, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[0, :], X[1, :], c=np.argmax(Y, axis=0), edgecolors='k', cmap=plt.cm.Paired)
    plt.title(f'Decision Boundary after {epoch} Epochs')
    plt.show()

db_plot(W, b, X, Y, epoch=3)
db_plot(W, b, X, Y, epoch=10)
db_plot(W, b, X, Y, epoch=100)
