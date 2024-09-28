# Re-import necessary libraries and re-define the simulation code for the Lorenz system

import numpy as np
import pandas as pd
import ace_tools as tools

# Define parameters for the Lorenz system
a = 10
b = 28
c = 8/3
dt = 0.01  # Time step
T = 10000  # Number of time steps

# Initialize variables for the Lorenz system
x = np.zeros(T)
y = np.zeros(T)
z = np.zeros(T)

# Set initial conditions for the Lorenz system
x[0] = 1.0
y[0] = 1.0
z[0] = 1.0

# Euler approximation for the Lorenz system
for t in range(T - 1):
    dx = a * (y[t] - x[t])
    dy = x[t] * (b - z[t]) - y[t]
    dz = x[t] * y[t] - c * z[t]
    
    x[t + 1] = x[t] + dt * dx
    y[t + 1] = y[t] + dt * dy
    z[t + 1] = z[t] + dt * dz

# Create a DataFrame to store the simulated data
lorenz_data = pd.DataFrame({'Time': np.arange(T) * dt, 'x': x, 'y': y, 'z': z})

# Split into training (first 70%) and test (last 30%) data
train_data_lorenz = lorenz_data.iloc[:int(0.7 * T)]
test_data_lorenz = lorenz_data.iloc[int(0.7 * T):]

# Display the training and test data to the user
tools.display_dataframe_to_user(name="Lorenz System Training Data", dataframe=train_data_lorenz)
tools.display_dataframe_to_user(name="Lorenz System Test Data", dataframe=test_data_lorenz)
