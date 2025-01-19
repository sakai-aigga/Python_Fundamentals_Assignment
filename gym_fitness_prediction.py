# -*- coding: utf-8 -*-
"""Gym_Fitness_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U9IqmaVGh1wG4JdAqZFOwZfwIlKoxERU
"""

# Importing all the libraries
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Importing the data file with link
data_url = "https://raw.githubusercontent.com/sayande01/Kaggle_Notebooks/refs/heads/main/gym_members_exercise_tracking.csv"
df = pd.read_csv(data_url)
df.head()

# Checking for null values
df.isnull().sum()

# Exploring the data
plt.scatter(df['Session_Duration (hours)'], df['Calories_Burned'])
plt.xlabel('Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('Calories Burned vs. Duration of session')
plt.show()

# We convert the data into pytorch tensors
X = df[['Session_Duration (hours)']]
y = df['Calories_Burned']

# We split the data into training and testing sets using the sklearn module we imported
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2509)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)  # Input size is 2 i.e. session duration, avg BPM

    def forward(self, x):
        x = self.layer1(x)
        return x

model = RegressionModel()
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.7)

train_loss=[]
test_loss=[]
train_accuracy=[]
train_loss=[]

num_epochs = 5000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    predicted_y = model(X_train_tensor).squeeze()
    y_train = y_train_tensor.squeeze()
    losses = loss(predicted_y, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    #printing loss in every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}')
# Evaluate the model
with torch.no_grad():
    predicted_y_test = model(X_test_tensor).squeeze()
    y_test=y_test_tensor.squeeze()
    test_loss = loss(predicted_y_test, y_test_tensor)

# Convert tensors to NumPy arrays before using with sklearn functions
# Else we get a Unsupported device for NumPy: device(type='cpu')
y_test_np = y_test.cpu().numpy()
predicted_y_test_np = predicted_y_test.cpu().numpy()

mse = mean_squared_error(y_test_np, predicted_y_test_np)
r2 = r2_score(y_test_np, predicted_y_test_np)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

plt.scatter(y_test, predicted_y_test)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='g')
plt.show()