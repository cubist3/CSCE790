import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = 10
b = 28
c = 8/3
dt = 0.01  
T = 10000  

x = np.zeros(T)
y = np.zeros(T)
z = np.zeros(T)

x[0] = 1.0
y[0] = 1.0
z[0] = 1.0

for t in range(T - 1):
    dx = a * (y[t] - x[t])
    dy = x[t] * (b - z[t]) - y[t]
    dz = x[t] * y[t] - c * z[t]

    x[t + 1] = x[t] + dt * dx
    y[t + 1] = y[t] + dt * dy
    z[t + 1] = z[t] + dt * dz

df_lorenz = pd.DataFrame({'Time': np.arange(T) * dt, 'x': x, 'y': y, 'z': z})

train_lorenz = df_lorenz.iloc[:int(0.7 * T)]
test_lorenz = df_lorenz.iloc[int(0.7 * T):]

plt.figure(figsize=(10, 6))
plt.plot(train_lorenz)
plt.title('Training Data - Lorenz System')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('C:/cubis/Desktop/HW/training_data_lorenz.eps', format='eps')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_lorenz)
plt.title('Test Data - Lorenz System')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('C:/cubis/Desktop/HW/test_data_lorenz.eps', format='eps')
plt.show()

a = 0.5
b = 2.0
c = 4.0
dt = 0.01
T = 10000

x = np.zeros(T)
y = np.zeros(T)
z = np.zeros(T)

x[0] = 1.0
y[0] = 1.0
z[0] = 1.0

for t in range(T - 1):
    dx = -y[t] - z[t]
    dy = x[t] + a * y[t]
    dz = b + z[t] * (x[t] - c)

    x[t + 1] = x[t] + dt * dx
    y[t + 1] = y[t] + dt * dy
    z[t + 1] = z[t] + dt * dz

data = pd.DataFrame({'Time': np.arange(T) * dt, 'x': x, 'y': y, 'z': z})

train_data = data.iloc[:int(0.7 * T)]
test_data = data.iloc[int(0.7 * T):]

plt.figure(figsize=(10, 6))
plt.plot(train_lorenz)
plt.title('Training Data - Rossler System')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('C:/cubis/Desktop/HW/training_data_rossler.eps', format='eps')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(test_lorenz)
plt.title('Test Data - Rossler System')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('C:/cubis/Desktop/HW/test_data_lorenz.eps', format='eps')
plt.show()

