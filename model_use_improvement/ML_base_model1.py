## This version uses the optimal macro parameters from the Baysian model

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the Excel file into a pandas dataframe
df = pd.read_excel('C:/Users/Colton/Desktop/bbg_test_data3.xlsx', 
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


# Train and evaluate the model for each optimizer
mse_scores = []

# Build the neural network model
model = Sequential()
model.add(Dense(16, input_dim=Xs, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with the current optimizer
model.compile(loss='mean_squared_error', optimizer='nadam')

# Train the model on the training data
model.fit(X_train, y_train, epochs=40, batch_size=7, verbose=0)

# Evaluate the model on the testing data and append the MSE to the scores list
mse = model.evaluate(X_test, y_test, verbose=0)
mse_scores.append(mse)

# Print the results for the current optimizer
avg_mse = np.mean(mse_scores)
print(f"Average MSE: {avg_mse}")