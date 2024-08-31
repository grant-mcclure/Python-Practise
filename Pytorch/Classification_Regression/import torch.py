import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Create Arrays
X = np.array([X for X in range(1000)])  # Create a 1D array of 1000 elements
X = X.reshape(-1, 1)  # Reshape into a column vector
y = 50 - 2 * X

plt.scatter(X, y, color='r', label='initial data')
plt.title('Pre Pytorch')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Normalize the data
x_mean, x_std = X.mean(), X.std()
y_mean, y_std = y.mean(), y.std()

X_normalised = (X - x_mean) / x_std
y_normalised = (y - y_mean) / y_std

x_tensor = torch.tensor(X_normalised, dtype=torch.float32)
y_tensor = torch.tensor(y_normalised, dtype=torch.float32)

plt.scatter(X_normalised, y_normalised, color='k', label='initial data')
plt.title('Normalized data Pre Pytorch')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Creating the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)  # Removed squeeze()

model = LinearRegressionModel(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=0.15)

# Set up training loop
num_epochs = 10
for epoch in range(num_epochs):
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
new_x = 50
new_x_normalised = (new_x - x_mean) / x_std
new_x_tensor = torch.tensor(new_x_normalised, dtype=torch.float32).reshape(-1, 1)

model.eval()
with torch.no_grad():
    prediction_normalised = model(new_x_tensor)

prediction_denormalised = prediction_normalised.item() * y_std + y_mean
print(f'Predicted value for x = {new_x}: {prediction_denormalised}')

# Plot the final data
plt.scatter(X, y, label='initial data')
fit_line = model(x_tensor).detach().numpy() * y_std + y_mean
plt.plot(X, fit_line, 'r', label='pytorch line')
plt.title('Pytorch with predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()  # Make sure to call show()
