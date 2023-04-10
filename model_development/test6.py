## this iteration tests different epochs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

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

# Build the neural network model
model = Sequential()
model.add(Dense(35, input_dim=Xs, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Define a list of epoch sizes to try
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Define a dictionary to store the MSE values for each optimizer
mse_dict = {'adam': [], 'adagrad': [], 'nadam': []}

# Train the model on each optimizer and epoch size
for optimizer in mse_dict.keys():
    for epochs in epochs_list:
        mse_list = []
        for i in range(3):
            # Compile the model with the current optimizer
            model.compile(loss='mean_squared_error', optimizer=optimizer)
            # Train the model on the training data
            model.fit(X_train, y_train, epochs=epochs, batch_size=10, verbose=0)
            # Evaluate the model on the testing data and store the MSE value
            mse = model.evaluate(X_test, y_test, verbose=0)
            mse_list.append(mse)
        # Calculate the average MSE and standard deviation for the current optimizer and epoch size
        avg_mse = np.mean(mse_list)
        std_mse = np.std(mse_list)
        # Add the average MSE to the dictionary
        mse_dict[optimizer].append(avg_mse)
        print(f"Optimizer: {optimizer}, Epochs: {epochs}, Mean Squared Error: {avg_mse}, Standard Deviation: {std_mse}")

# Plot the MSE values for each optimizer
plt.figure()
for optimizer, mse_values in mse_dict.items():
    plt.plot(epochs_list, mse_values, label=optimizer)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
