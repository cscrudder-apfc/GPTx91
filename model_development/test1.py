## V1

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Load the Excel file into a pandas dataframe
df = pd.read_excel('C:/Users/Colton/Desktop/bbg_test_data.xlsx', 
                   usecols=['Duration', 'Composite_Level', 'Z-Sprd'])

# Clean the data by removing rows with missing or invalid values
df = df.dropna()
df = df[np.isfinite(df['Duration'])]
df = df[df['Composite_Level'].between(0, 10, inclusive='both')]

# Define the input and output variables
X = df[['Duration', 'Composite_Level']]
y = df['Z-Sprd']

# Build the neural network model
model = Sequential()
model.add(Dense(20, input_dim=2, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=100, batch_size=10, verbose=0)

# Evaluate the model on the training data
mse = model.evaluate(X, y, verbose=0)
print(f"Mean squared error on training data: {mse}")
