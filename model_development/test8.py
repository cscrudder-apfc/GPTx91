import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

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

# Define the search space for Bayesian optimization
dimensions = [
    Real(low=0.0001, high=0.1, name='learning_rate'),
    Categorical(categories=['relu', 'tanh'], name='activation'),
    Real(low=0.01, high=1.0, name='dropout_rate'),
    Categorical(categories=[16, 32, 64, 128, 256], name='batch_size')
]

# Define the objective function to minimize (i.e., MSE)
@use_named_args(dimensions=dimensions)
def objective(learning_rate, activation, dropout_rate, batch_size):
    # Build the neural network model
    model = Sequential()
    model.add(Dense(35, input_dim=Xs, activation=activation))
    model.add(Dense(20, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(1, activation='linear'))
    
    # Compile the model with the current hyperparameters
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mse'])
    
    # Train the model on the training data with the current batch size
    model.fit(X_train, y_train, epochs=40, batch_size=batch_size, verbose=0)
    
    # Evaluate the model on the testing data and return the MSE
    _, mse = model.evaluate(X_test, y_test, verbose=0)
    
    return mse

# Run Bayesian optimization to find the best hyperparameters
res = gp_minimize(objective, dimensions, n_calls=50, random_state=42)

# Print the best hyperparameters and corresponding MSE
print(f"Best hyperparameters: {res.x}")
print(f"Corresponding MSE: {res.fun}")
