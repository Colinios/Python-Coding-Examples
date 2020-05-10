#Naive Bayes classification: Email detection

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Build model to distinguish between emails about soccer or hockey
train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

# Counting the words
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Making a Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))
#The model has a fitting score of 97.2%, having a very high explanatory value



# Alternative datasets to distinguish between emails about soccer or PC hardware
train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware','rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))
# Almost perfect fitting score of 99.7%