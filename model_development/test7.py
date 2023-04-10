## this iteration tests different batch sizes

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the Excel file into a pandas dataframe
df = pd.read_excel('C:/Users/Colton/Desktop/bbg_test_data.xlsx', 
                   usecols=['Duration', 'Composite_Level', 'Country_Risk', 'Z-Sprd','Classification'])

# Clean the data by removing rows with missing or invalid values
df = df.dropna()

# Encode the categorical variable "Country_Risk"
Country_Risk_dummies = pd.get_dummies(df['Country_Risk'], prefix='Country_Risk')
df = pd.concat([df, Country_Risk_dummies], axis=1)
df = df.drop(columns=['Country_Risk']) # drop original column

# Encode the categorical variable "Classification"
Classification_dummies = pd.get_dummies(df['Classification'], prefix='Classification')
df = pd.concat([df, Classification_dummies], axis=1)
df = df.drop(columns=['Classification']) # drop original column

# Define the input and output variables
headers_list = list(df.columns)
headers_list.remove('Z-Sprd')

X = df[headers_list]
Xs = len(headers_list)
y = df['Z-Sprd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the batch sizes to test
batch_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Create lists to store the MSE results for each batch size
train_mse_list = []
test_mse_list = []

# Train the model for each batch size
for batch_size in batch_sizes:
    print(f"Training model with batch size {batch_size}")
    
    # Build the neural network model
    model = Sequential()
    model.add(Dense(35, input_dim=Xs, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='nadam')

    # Train the model on the training data
    history = model.fit(X_train, y_train, epochs=40, batch_size=batch_size, verbose=0)

    # Evaluate the model on the training data
    train_mse = model.evaluate(X_train, y_train, verbose=0)
    print(f"Mean squared error on training data: {train_mse}")
    train_mse_list.append(train_mse)

    # Evaluate the model on the testing data
    test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean squared error on testing data: {test_mse}")
    test_mse_list.append(test_mse)

# Plot the MSE results for each batch size
plt.plot(batch_sizes, train_mse_list, label='Train MSE')
plt.plot(batch_sizes, test_mse_list, label='Test MSE')
plt.xlabel('Batch Size')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Batch Size')
plt.legend()
plt.show()
