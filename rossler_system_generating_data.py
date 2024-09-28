import numpy as np
import pandas as pd

# Define parameters for the Rossler system
a = 0.5
b = 2.0
c = 4.0
dt = 0.01  # Time step
T = 10000  # Number of time steps

# Initialize variables
x = np.zeros(T)
y = np.zeros(T)
z = np.zeros(T)

# Set initial conditions
x[0] = 1.0
y[0] = 1.0
z[0] = 1.0

# Euler approximation for the Rossler system
for t in range(T - 1):
    dx = -y[t] - z[t]
    dy = x[t] + a * y[t]
    dz = b + z[t] * (x[t] - c)
    
    x[t + 1] = x[t] + dt * dx
    y[t + 1] = y[t] + dt * dy
    z[t + 1] = z[t] + dt * dz

# Create a DataFrame to store the simulated data
data = pd.DataFrame({'Time': np.arange(T) * dt, 'x': x, 'y': y, 'z': z})

# Split into training (first 70%) and test (last 30%) data
train_data = data.iloc[:int(0.7 * T)]
test_data = data.iloc[int(0.7 * T):]

# Display the training and test data to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="Rossler System Training Data", dataframe=train_data)
tools.display_dataframe_to_user(name="Rossler System Test Data", dataframe=test_data)
