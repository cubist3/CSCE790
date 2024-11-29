# -*- coding: utf-8 -*-
# Part B
# Q2

# Y seems continuous so used linear in the output layer

def relu(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return (Z > 0).astype(float)

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
# W1, W2 - adding more neurons (100) to the hidden layer
W1 = np.random.randn(100, 1) * np.sqrt(1 / 1)
W2 = np.random.randn(1, 100) * np.sqrt(1 / 100)
b1 = np.zeros((100, 1))
b2 = np.zeros((1, 1))

eta = 0.1
epochs = 2000

# normalising input and output
X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

error_TRs = []

for epoch in range(epochs):
    Z1 = np.dot(W1, X) + b1
    pred1 = relu(Z1)

    Z2 = np.dot(W2, pred1) + b2
    pred_y = linear(Z2)

    error = pred_y - Y
    error_TR = np.mean(np.square(error))
    error_TRs.append(error_TR)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Training Error: {error_TR:.6f}")

    dZ2 = error * linear_deriv(Z2)
    dW2 = np.dot(dZ2, pred1.T) / X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

    dpred1 = np.dot(W2.T, dZ2)
    dZ1 = dpred1 * relu_deriv(Z1)
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True) / X.shape[1]

    W2 -= eta * dW2
    b2 -= eta * db2
    W1 -= eta * dW1
    b1 -= eta * db1

plt.plot(range(epochs), error_TRs)
plt.xlabel('Epochs')
plt.ylabel('Training Error (MSE)')
plt.title('Training Error vs Epochs')
plt.show()

# final output
Z1 = np.dot(W1, X) + b1
pred1 = relu(Z1)
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

len(error_TRs)

epochs

print(error_TRs)

Z1 = np.dot(W1, X) + b1
pred1 = relu(Z1)
Z2 = np.dot(W2, pred1) + b2
Y_pred = linear(Z2)

plt.figure()
plt.plot(X.flatten(), Y.flatten(), 'o-', label='Actual Function')
plt.plot(X.flatten(), Y_pred.flatten(), '.-', label='NN Approximation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Actual vs. NN Approximation (Final)')
plt.show()

save_path = '/content/drive/MyDrive/Colab Notebooks/actual_versus_NN.eps'

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)

def nn_approx(W1, b1, W2, b2, epoch):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    Y_pred = linear(Z2)

    plt.figure()
    plt.plot(X.flatten(), Y.flatten(), 'o-', label='Actual Function')
    plt.plot(X.flatten(), Y_pred.flatten(), 'x-', label=f'NN Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'NN Approximation after {epoch} Epochs')
    plt.show()

save_path = '/content/drive/MyDrive/Colab Notebooks/NN_approx_10_epochs.eps'
plt.figure()
nn_approx(W1, b1, W2, b2, epoch=10)
plt.savefig(save_path, format='eps')
plt.close()

save_path = '/content/drive/MyDrive/Colab Notebooks/NN_approx_100_epochs.eps'
plt.figure()
nn_approx(W1, b1, W2, b2, epoch=100)
plt.savefig(save_path, format='eps')
plt.close()

save_path = '/content/drive/MyDrive/Colab Notebooks/NN_approx_200_epochs.eps'
plt.figure()
nn_approx(W1, b1, W2, b2, epoch=200)
plt.savefig(save_path, format='eps')
plt.close()

save_path = '/content/drive/MyDrive/Colab Notebooks/NN_approx_400_epochs.eps'
plt.figure()
nn_approx(W1, b1, W2, b2, epoch=400)
plt.savefig(save_path, format='eps')
plt.close()

save_path = '/content/drive/MyDrive/Colab Notebooks/NN_approx_1000_epochs.eps'
plt.figure()
nn_approx(W1, b1, W2, b2, epoch=1000)
plt.savefig(save_path, format='eps')
plt.close()
