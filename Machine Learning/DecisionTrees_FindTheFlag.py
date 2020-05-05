import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


flags = pd.read_csv('flags.csv', header = 0)
print(flags.columns)
print(flags.head())


labels = flags[['Area']]
#First dataset only with colors. 
#data = flags[['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange']]

#Added the shape features in a later step to the data, since previous features showed little effect on the Landmass variable
data = flags[['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange', 'Circles', 'Crosses', 'Saltires', 'Quarters', 'Sunstars', 'Crescent', 'Triangle']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

scores = []
for i in range(1, 21):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))

plt.plot(range(1, 21), scores)
plt.xlabel('Depth')
plt.ylabel('R2')
plt.title('Fit of model vs. different depths of tree')
plt.show()
# The best fit for predicting the Landmass of a country is by using a tree depth of 5 features