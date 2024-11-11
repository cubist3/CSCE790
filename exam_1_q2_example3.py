# -*- coding: utf-8 -*-
# Example 3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# NN for the function f
class fNN(nn.Module):
    def __init__(self):
        super(fNN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        self.apply(self.init_weights)  

    def forward(self, y_p):
        x = torch.relu(self.fc1(y_p))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

# NN for the function g
class gNN(nn.Module):
    def __init__(self):
        super(gNN, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        self.apply(self.init_weights) 

    def forward(self, u):
        x = torch.relu(self.fc1(u))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

def true_f(y_p):
    return y_p / (1 + y_p**2)

def true_g(u):
    return u**3

net_f = fNN()
net_g = gNN()
f_optim = optim.SGD(net_f.parameters(), lr=0.1, momentum=0) 
g_optim = optim.SGD(net_g.parameters(), lr=0.1, momentum=0)
criterion = nn.MSELoss()

k_steps = 100000
u_input = np.random.uniform(-2, 2, k_steps) 
y_true = [0, 0]
y_pred = [0, 0]

# we want k+1 output as a current so k-th is [-1] and so forth..., one-step back
for k in range(2, k_steps):
    y_p_tensor = torch.tensor([[y_true[-1]]], dtype=torch.float32)
    u_tensor = torch.tensor([[u_input[k]]], dtype=torch.float32)

    y_true_k1 = true_f(y_true[-1]) + true_g(u_input[k])
    y_true.append(y_true_k1)

    f_pred = net_f(y_p_tensor)
    g_pred = net_g(u_tensor)
    y_pred_k1 = f_pred + g_pred
    y_pred.append(y_pred_k1.item())

    f_optim.zero_grad()
    g_optim.zero_grad()
    loss = criterion(y_pred_k1, torch.tensor([[y_true_k1]], dtype=torch.float32))
    loss.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(net_f.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)

    f_optim.step()
    g_optim.step()

    # print every 1000 steps
    if k % 1000 == 0:
        print(f"Step {k}/{k_steps}, Loss: {loss.item()}")

# figure (a)
y_vals = np.linspace(-10, 10, 200)
true_f_vals = true_f(y_vals)  # Calculate true f(y) values
f_NN_vals = [net_f(torch.tensor([[y]], dtype=torch.float32)).item() for y in y_vals]
plt.figure(figsize=(8, 6))
plt.plot(y_vals, true_f_vals, label='f(y)', color='blue', linestyle='-')
plt.plot(y_vals, f_NN_vals, label='estimate of f(y)', color='orange', linestyle='--')
plt.title("(a) Plots of the functions f and f_hat")
plt.ylabel("f(y) and f_hat (NN estimate)")
plt.legend()
plt.show()

# figure (b)
u_vals = np.linspace(-2, 2, 200)
true_g_vals = true_g(u_vals)
g_NN_vals = [net_g(torch.tensor([[u]], dtype=torch.float32)).item() for u in u_vals]  # Model approximation

plt.figure(figsize=(8, 6))
plt.plot(u_vals, true_g_vals, label='g(u)', color='blue', linestyle='-')
plt.plot(u_vals, g_NN_vals, label='estimate of g(u)', color='orange', linestyle='--')
plt.title("(b) Plots of the functions g and g_hat")
plt.xlabel("u")
plt.ylabel("g(u) and g_hat (NN estimate)")
plt.legend()
plt.show()

# figure (c) training
num_steps = 500
u_test_input = np.sin(2 * np.pi * np.arange(0, num_steps) / 25) + np.sin(2 * np.pi * np.arange(0, num_steps) / 10)
y_true_test = [0, 0]
y_pred_test = [0, 0]

for k in range(2, num_steps):
    y_p_tensor = torch.tensor([[y_true_test[-1]]], dtype=torch.float32)
    u_tensor = torch.tensor([[u_test_input[k]]], dtype=torch.float32)

    y_true_k1 = true_f(y_true_test[-1]) + true_g(u_test_input[k])
    y_true_test.append(y_true_k1)

    f_pred = net_f(y_p_tensor)
    g_pred = net_g(u_tensor)
    y_pred_k1 = f_pred + g_pred
    y_pred_test.append(y_pred_k1.item())

# figure (c)
plt.figure(figsize=(10, 5))
plt.plot(range(100), y_true_test[:100], label="f", color='black')
plt.plot(range(100), y_pred_test[:100], label="estimate of f", color='red', linestyle='dashed')
plt.title("(c) Outputs of the plant and the identification model")
plt.ylabel("f(y) and f_hat (NN estimate)")
plt.legend()
plt.show()

