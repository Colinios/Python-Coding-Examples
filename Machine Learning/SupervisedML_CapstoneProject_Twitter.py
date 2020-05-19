#Supervised Machine Learning Capstone: Twitter

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#PART 1		
#Find the most viral tweets using K-Nearest Neighbor algorithm

all_tweets = pd.read_json("random_tweets.json", lines=True)

print(len(all_tweets))
print(all_tweets.columns)
print(all_tweets.loc[0]['text'])

#Printing the first user and his location
print(all_tweets.loc[0]["user"])
print(all_tweets.loc[0]["user"]["location"])


#Defining Viral Tweets
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > np.median(all_tweets["retweet_count"]), 1, 0)
print(all_tweets['is_viral'].value_counts())

#Making features
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

#Normalizing the data
labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count', 'friends_count']]
scaled_data = scale(data, axis = 0)
print(scaled_data[0])

#Creating the Training Set and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size = 0.2, random_state = 1)

#Using the classifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(train_data, train_labels)
print(classifier.score(test_data, test_labels))
#Rather low fitting with an R2 of 0.59

#Finding the optimal k
import matplotlib.pyplot as plt
scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_data, train_labels)
    scores.append(classifier.score(test_data, test_labels))
plt.plot(range(1, 200), scores)
plt.show()
#We can see the highest R2 scores in the k-range lie between 25 and 75

#Using the classifier with k = 50
classifier = KNeighborsClassifier(n_neighbors = 50)
classifier.fit(train_data, train_labels)
print(classifier.score(test_data, test_labels))
#Improved with 0.62, but still not great


#PART 2
# Creating a classification algorithm that can classify any tweet (or sentence) and predict whether that sentence came from New York, London, or Paris

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#Investigate the datasets New York, London and Paris
new_york_tweets = pd.read_json("new_york.json", lines=True)
print(len(new_york_tweets))
print(new_york_tweets.columns)
print(new_york_tweets.loc[12]["text"])

london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)
print(len(london_tweets))
print(len(paris_tweets))

#Classifying using language: Naive Bayes Classifier
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()


all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

#Making a Training and Test Set
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size = 0.2, random_state = 1)
print(len(train_data))
print(len(test_data))

#Making the Count Vectors
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

print(train_data[3])
print(train_counts[3])

#Train and Test the Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

#Evaluating Your Model via accuracy score (and confusion matrix)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, predictions))
#Accuracy lies around 67.8%, which is ok, but not great

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_labels, predictions))


#Testing the prediction function with own tweet
tweet = 'Pierre, the baquette is tasty'
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))