# 2(a) perceptron with 2 inputs and 1 output
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def domain(samples):
    x1 = np.linspace(-2, 2, samples)
    x2 = np.linspace(-2, 2, samples)
    return np.meshgrid(x1, x2)

def perceptron_output(x1, x2, weights, bias):
    return weights[0] * x1 + weights[1] * x2 + bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hard_limit(z):
    return np.where(z >= 0, 1, 0)

def radial_basis(z):
    return np.exp(-z**2)

weights = [-4.79, 5.90]
bias = -0.93

def plot_surface(X1, X2, Y, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()
    
def plot_final(samples):
    X1, X2 = domain(samples)
    z = perceptron_output(X1, X2, weights, bias)
    y_sigmoid = sigmoid(z)
    plot_surface(X1, X2, y_sigmoid, f"Sigmoid Activation : {samples}")
    plt.savefig(f'plot_final_{samples}.eps', format='eps')
    y_hard_limit = hard_limit(z)
    plot_surface(X1, X2, y_hard_limit, f"Hard Limit Activation : {samples}")
    plt.savefig(f'plot_final_{samples}.eps', format='eps')
    y_rbf = radial_basis(z)
    plot_surface(X1, X2, y_rbf, f"Radial Basis Function Activation : {samples}")
    plt.savefig(f'plot_final_{samples}.eps', format='eps')

plot_final(100)
plot_final(5000)
plot_final(10000)

# (b) 2-layer NN with 2 inputs and 1 output
def two_layer_nn(x1, x2, activation_func):
    V_T = np.array([[-2.69, -2.80],
                    [-3.39, -4.56]])
    bv = np.array([-2.21, 4.76])
    W = np.array([-4.91, 4.95])
    bw = -2.28

    x = np.array([x1, x2])
    z = np.dot(V_T, x) + bv
    a = activation_func(z)
    y = np.dot(W, a) + bw

    return y

def output_surface(activation_func, num_points):
    x1_vals = np.linspace(-2, 2, num_points)
    x2_vals = np.linspace(-2, 2, num_points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    Y = np.zeros_like(X1)
    # 2 layers - consider it as a composite function
    for i in range(num_points):
        for j in range(num_points):
            Y[i, j] = two_layer_nn(X1[i, j], X2[i, j], activation_func)
    return X1, X2, Y

sample_list = [100, 5000, 10000]

for sample in sample_list:
    X1, X2, Y_sigmoid = output_surface(sigmoid, sample)
    plot_surface(X1, X2, Y_sigmoid, f'Sigmoid Activation Function - {sample}')
    plt.savefig(f'sigmoid_activation_{sample}.eps', format='eps')
    plt.close() 

    X1, X2, Y_hard_limit = output_surface(hard_limit, sample)
    plot_surface(X1, X2, Y_hard_limit, f'Hard Limit Activation Function - {sample}')
    plt.savefig(f'hard_limit_activation_{sample}.eps', format='eps')
    plt.close() 

    X1, X2, Y_radial_basis = output_surface(radial_basis, sample)
    plot_surface(X1, X2, Y_radial_basis, f'Radial Basis Function - {sample}')
    plt.savefig(f'radial_basis_activation_{sample}.eps', format='eps')
    plt.close()
