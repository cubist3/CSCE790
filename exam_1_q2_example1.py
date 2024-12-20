# -*- coding: utf-8 -*-
# Example 1

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def true_f(u):
    return 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u)

# NN model N[u(k)] class
class identificationNN(nn.Module):
    def __init__(self):
        super(identificationNN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)  # 1,20,10,1 class model

    def forward(self, u):
        x = torch.relu(self.fc1(u))  # applying ReLU after 1st layer
        x = torch.relu(self.fc2(x))  # applying ReLU after 2nd layer
        return self.fc3(x)           # output, final output

# an alternative NL function after identification (provided in problem)
def alternative_f(u):
    return u**3 + 0.3 * u**2 - 0.4 * u

# gradient optimization method
net = identificationNN()
optimizer = optim.SGD(net.parameters(), lr=0.25, momentum=0)  # Step size (learning rate) set to 0.25 as specified
criterion = nn.MSELoss()

# (a): sinusoidal input until k=500
k = 500
u_sin_input = np.sin(2 * np.pi * np.arange(0, k) / 250)
y_true_a = [0, 0]  # initial conditions - true
y_pred_a = [0, 0]  # Initial conditions - pred

for k in range(2, k):
    u = torch.tensor([[u_sin_input[k]]], dtype=torch.float32, requires_grad=True)

    # true output based on the true function f(u)
    y_true_k1 = 0.3 * y_true_a[-1] + 0.6 * y_true_a[-2] + true_f(u.item())
    y_true_a.append(y_true_k1)

    # NN model output (pred) based on the NN (net(u))
    y_pred_k1 = 0.3 * y_pred_a[-1] + 0.6 * y_pred_a[-2] + net(u)
    y_pred_a.append(y_pred_k1.item())

    # loss calculation with optimizer
    optimizer.zero_grad()
    loss = criterion(y_pred_k1, torch.tensor([[y_true_k1]], dtype=torch.float32))
    loss.backward()
    optimizer.step()

plt.figure(figsize=(10, 5))
plt.plot(range(100, 500), y_true_a[100:500], label="True")
plt.plot(range(100, 500), y_pred_a[100:500], label="Model Output Pred (NN)", linestyle='dashed')
plt.title("(a) Outputs of the plant and identification model when adaptation stops at k=500")
plt.xlabel("steps")
plt.ylabel("y_p or y_p_hat")
plt.legend()
plt.show()

# (b): training with a random input - unif over 50,000 steps
k_rand = 50000
u_random_input = np.random.uniform(-1, 1, k_rand)
y_true_b = [y_true_a[-2], y_true_a[-1]]  # start from the end period of (a)
y_pred_b = [y_pred_a[-2], y_pred_a[-1]]

for k in range(k_rand):
    u = torch.tensor([[u_random_input[k]]], dtype=torch.float32, requires_grad=True)

    # true value with an alternative NL func for the identification
    y_true_k1 = 0.3 * y_true_b[-1] + 0.6 * y_true_b[-2] + alternative_f(u.item())
    y_true_b.append(y_true_k1)

    # NN model output (pred) based on the NN (net(u))
    y_pred_k1 = 0.3 * y_pred_b[-1] + 0.6 * y_pred_b[-2] + net(u)
    y_pred_b.append(y_pred_k1.item())

    # loss calculation with optimizer
    optimizer.zero_grad()
    loss = criterion(y_pred_k1, torch.tensor([[y_true_k1]], dtype=torch.float32))
    loss.backward()
    optimizer.step()

plt.figure(figsize=(10, 5))
plt.plot(range(500), y_true_b[:500], label="True")
plt.plot(range(500), y_pred_b[:500], label="Model Output Pred (NN)", linestyle='dashed')
plt.title("(b) Response of plant and identification model after identification using a random input")
plt.xlabel("steps")
plt.ylabel"y_p or y_p_hat")
plt.legend()
plt.show()