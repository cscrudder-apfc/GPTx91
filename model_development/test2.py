## V2

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the Excel file into a pandas dataframe
df = pd.read_excel('C:/Users/Colton/Desktop/bbg_test_data.xlsx', 
                   usecols=['Duration', 'Composite_Level', 'Country_Risk', 'Z-Sprd','Classification'])

# Clean the data by removing rows with missing or invalid values
df = df.dropna()


# Check for NaN values in the dataframe and replace them with 0
# if df.isna().sum().sum() > 0:
#     df = df.fillna(0)

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

# Build the neural network model
model = Sequential()
model.add(Dense(50, input_dim=Xs, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=100, batch_size=10, verbose=0)

# Evaluate the model on the training data
mse = model.evaluate(X, y, verbose=0)
print(f"Mean squared error on training data: {mse}")

# Predict on the training data
y_pred = model.predict(X)

# Add the predicted values to the original dataframe
df['Predicted_Z-Sprd'] = y_pred

# Write the updated dataframe to a new Excel file
writer = pd.ExcelWriter('C:/Users/Colton/Desktop/bbg_test_data_with_predictions.xlsx', engine='xlsxwriter')
df.to_excel(writer, index=False)
writer.save()

print("Predictions saved to file.")
