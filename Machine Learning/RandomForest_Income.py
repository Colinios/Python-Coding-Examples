RandomForest_Income

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv('income.csv', header = 0, delimiter = ', ')
print(income_data.iloc[0])

#Formatting data for random forest model
labels = income_data[['income']]
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

#Creating the random forest
forest = RandomForestClassifier(random_state = 1)
forest.fit(train_data, train_labels)
predictions = forest.predict(test_data)
print(forest.score(test_data, test_labels))
#The model shows an accuracy of 0.82, which is acceptable