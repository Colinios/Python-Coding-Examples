### Logistic Regression: Titanic model ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Data from https://www.kaggle.com/c/titanic
passengers = pd.read_csv('passengers.csv')
print(passengers.head())
passengers.info()

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'female':'1', 'male':'0'})


print(passengers.Age.values)
# Fill the nan values in the age column
passengers['Age'].fillna(value = passengers['Age'].mean(), inplace = True)

# Create first and second class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 2 if x == 2 else 0)
#print(passengers.head())

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
train_features, test_features, train_labels, test_labels =  train_test_split(features, survival, train_size=0.8, test_size=0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Scoring the model on the train data
print(model.score(train_features, train_labels))

# Scoring the model on the test data
print(model.score(test_features, test_labels))

# Analyzing the coefficients
print(model.coef_)
print(model.intercept_)

# Defining Jack and Rose passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])

# Combining passenger arrays
sample_passengers = np.array([Jack, Rose])

# Scaling the sample passenger features
sample_passengers = scaler.transform(sample_passengers )
print(sample_passengers)

# Making survival predictions
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))
#Rose is predicted to survive with a 95% probability
#Jack on the opposite only has a 11% probability of surviving
#Main difference is the sex and first class