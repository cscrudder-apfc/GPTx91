import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the Excel file into a pandas dataframe
df = pd.read_excel('C:/Users/Colton/Desktop/bbg_test_data2.xlsx', 
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

# Remove rows with missing or invalid values from the test set
X_test = X_test.dropna()
y_test = y_test[X_test.index]

# Train the model and predict the Z-Spread for the test data
model = Sequential()
model.add(Dense(16, input_dim=Xs, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='nadam')
model.fit(X_train, y_train, epochs=40, batch_size=7, verbose=0)
y_pred = model.predict(X_test)

# Add a new column "Predicted_Z-Sprd" to the Excel file with the predicted Z-Spread values
df_pred = pd.DataFrame(y_pred, columns=['Predicted_Z-Sprd'])
df_test = pd.DataFrame(X_test, columns=headers_list)
df_test.reset_index(drop=True, inplace=True)
df_final = pd.concat([df_test, df_pred], axis=1)
df_final.to_excel('C:/Users/Colton/Desktop/bbg_test_data2_predicted.xlsx', index=False)
