import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

concrete_data = pd.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv')
concrete_data.head()
concrete_data.shape	# About 1000 samples available to train the model (careful of not overfitting)

#Check for missing values in dataset
concrete_data.describe()
concrete_data.isnull().sum()	# No issue with missing values

# Splitting data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

#Normalizing the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()
n_cols = predictors_norm.shape[1]

#Building the neural network

def regression_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#Training and Testing the model
model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)