Yelp Multiple Regression Analysis
#Dataset not for public use, therefore not attached for code review

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Import data sets
businesses = pd.read_json('yelp_business.json', lines=True)
reviews = pd.read_json('yelp_review.json', lines=True)
users = pd.read_json('yelp_user.json', lines=True)
checkins = pd.read_json('yelp_checkin.json', lines=True)
tips = pd.read_json('yelp_tip.json', lines=True)
photos = pd.read_json('yelp_photo.json', lines=True)

pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500

#How many different businesses are in the dataset? What are the different features in the review DataFrame?
print(len(businesses))
print(reviews.columns)
users.describe

#Merging the 6 datasets to one with left join
df = pd.merge(businesses, photos, how='left', on='business_id')
df = pd.merge(businesses, users, how='left', on='business_id')
df = pd.merge(businesses, checkins, how='left', on='business_id')
df = pd.merge(businesses, tips, how='left', on='business_id')
df = pd.merge(businesses, photos, how='left', on='business_id')

#Cleaning the data
#Removing variables which are not continious or binary
features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state', 'time']
df.drop(labels=features_to_remove, axis=1, inplace=True)

#Checking data for missing values and fill them with 0
df.isna().any()
df.fillna({'weekday_checkins': 0,
           'weekend_checkins': 0,
           'number_tips': 0,
           'average_caption_length': 0,
           'number_pics ': 0,
           'average_tip_length': 0},
          inplace=True)
df.isna().any()

#Data exploration
#Which variables have a high correlation with the stars (rating) variable?
df.corr()
#Especially the average review sentiment had the largest (positive) correlation,
#besides average review lenght and age

#Plotting for visualization
plt.scatter(df['average_review_sentiment'],df['stars'],alpha=0.1)
plt.xlabel('average_review_sentiment')
plt.ylabel('Yelp Rating')
plt.show()



## 1st (simpler) model with variables Avg review length and age ##

#Seperate out our features
features = df[['average_review_length','average_review_age']]
ratings = df['stars']

#Split the data into training and testing sets and create model
x_train, x_test, y_train, y_test = train_test_split(ratings, features, train_size=0.8, test_size=0.2, random_state = 1)
model = LinearRegression()
model.fit(x_train,y_train)

#Evaluate the model with R^2
model.score(x_train, y_train)
model.score(x_test, y_test)
#0.083 and 0.081 resp. explain a low proportion of the depend variables (star) variance,
#Therefore they are not that useful

sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)

y_predicted = model.predict(X_test)
plt.scatter(y_test, y_predicted, alpha = 0.3)
plt.xlabel('Yelp Rating')
plt.ylabel('Predicted Yelp Rating')
plt.ylim(1,5)
plt.show()
#confirms our results of low correlation



## 2nd model including the variable avg review sentiment ##

feature_subset = ['average_review_length','average_review_age', 'average_review_sentiment']

def model_these_features(feature_list):
        ratings = df.loc[:,'stars']
    features = df.loc[:,feature_list]
    
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    if len(X_train.shape) < 2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    y_predicted = model.predict(X_test)
    
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()

model_these_features(feature_subset)
#Improved correlation of 0.64, eplaining a good amount of the variation in rating

