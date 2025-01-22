
# Importing all the libraries
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Importing the data file
data_url = "https://raw.githubusercontent.com/sayande01/Kaggle_Notebooks/refs/heads/main/gym_members_exercise_tracking.csv"
df = pd.read_csv(data_url)
df.head()

#for number of rows and columns
df.shape

# Checking for null values
df.isnull().sum()

# Visualizing the data and I used only two features i.e session duration and calories burned
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

#Training the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)  # Input size is 1 for session duration

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

num_epochs = 6000
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

# Making predictions with the model
y_test_np = y_test.cpu().numpy()
predicted_y_test_np = predicted_y_test.cpu().numpy()

plt.scatter(y_test, predicted_y_test)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs Predicted values")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='g')
plt.show()

# Evaluating the model accuracy
mse = mean_squared_error(y_test_np, predicted_y_test_np)
r2 = r2_score(y_test_np, predicted_y_test_np)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Handling user inputs using the model we trained
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler on the training data4


while True:
    try:
        user_input = float(input("Enter the session duration (in hours): "))
        user_input_tensor = torch.tensor([[user_input]], dtype=torch.float32)#converts user input to a 2D tensor
        user_input_tensor = scaler.transform(user_input_tensor) #scales the user input into standard scaler as expected by model
        user_input_tensor = torch.tensor(user_input_tensor, dtype=torch.float32) #converts the scaled input back into tensor

        with torch.no_grad():
            predicted_calories = model(user_input_tensor).item() #passing the tensor into the trained model

        print(f"Predicted calories burned: {predicted_calories:.2f}") # Prints the output with 2 decimals

    except ValueError:
        print("Invalid input. Please enter a number for session duration.")

    another_prediction = input("Do you want to make another prediction? (y/n): ")
    if another_prediction.lower() != 'y':
        break
