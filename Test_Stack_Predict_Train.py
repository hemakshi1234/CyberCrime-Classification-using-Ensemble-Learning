# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:59:40 2020

@author: Hemakshi Pandey
"""
# Implementing voting ensemble with bag of words model


## Importing the libraries

import pandas as pd   
#They are used in Python to deal with data analysis and manipulation. To put it in simpler words, Pandas help us to organize data and manipulate the data by putting it in a tabular form.

import nltk
# NLTK is a leading platform for building Python programs to work with human language data.

import pickle 
#Comes handy to save complicated data.Python pickle module is used for serializing and de-serializing python object structures.

import re
#This module provides regular expression matching operations

from nltk.corpus import stopwords
nltk.download('stopwords')
# One of the major forms of pre-processing is to filter out useless data. 
#In natural language processing, useless words (data), are referred to as stop words.


nltk.download('wordnet')
wnlem = nltk.WordNetLemmatizer()
#Lemmatization, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language.

nltk.download('punkt')
#This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.

"""## Importing the dataset"""

dataset = pd.read_csv('FINAL_DATA_22_9_2020_IT_Act_2000.csv')  # This data contains the labelled definitions of IPC 302,307 and 376

dataset.shape

dataset["Label"].value_counts()

dataset.head()     #The head() function is used to get the first n rows.

"""## Cleaning the texts"""

corpus = []     # defining a list of corpus
for i in range(0, 288):  # the loop for traversing through the rows
  definition = re.sub('[^a-zA-Z]', ' ', dataset['Definition'][i])     # the operation takes input of all word including alphabet
  definition = definition.lower()     # converts that into lower case (normalization and cleaning)
  definition = definition.split()   #split() method returns a list of strings after breaking the given string by the specified separator.
  wnlem = nltk.WordNetLemmatizer()  #brings context to the words.
  porter = nltk.PorterStemmer()
  lancaster = nltk.LancasterStemmer() 
  all_stopwords = stopwords.words('english') #useless words (data), are referred to as stop words.
  definition = [wnlem.lemmatize(word) for word in definition if not word in set(all_stopwords)]  # traversing through the words and normalizing it 
  #definition = [porter.stem(word) for word in definition if not word in set(all_stopwords)]
  #definition = [lancaster.stem(word) for word in definition if not word in set(all_stopwords)]
  definition = ' '.join(definition)     #Join all items in a tuple into a string, using a space (' ') character as separator:
  corpus.append(definition)            # filtered definition are added to the list

print(corpus)

"""## Creating the Bag of Words model"""

from sklearn.feature_extraction.text import CountVectorizer  #Convert a collection of text words to a matrix of token counts
cv = CountVectorizer(max_features = 840 )

#max_features = 450
#With CountVectorizer we are converting raw text to a numerical vector representation of words. 
#This makes it easy to directly use this representation as features in Machine Learning tasks such as for text classification and clustering.
X = cv.fit_transform(corpus).toarray() #one step fit tranform
#Here the fit method, when applied to the training dataset,learns the model parameters (for example, mean and standard deviation). 
#We then need to apply the transform method on the training dataset to get the transformed (scaled) training dataset.
y = dataset.iloc[:, -1].values

len(X[0])

print(y)

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
# define example
values = array(y)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
print(integer_encoded)
y1 = integer_encoded
inverted = label_encoder.inverse_transform([argmax(integer_encoded[0])])
print(inverted)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.20, stratify = y1)

"""## Constructing Model Stacking Ensemble Classifier"""

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

M1 = DecisionTreeClassifier()
M2 = SVC(kernel = 'linear', random_state = 0, probability=True)
M3 = LogisticRegression()

M1.fit(X_train, y_train)
M2.fit(X_train, y_train)
M3.fit(X_train, y_train)

print(M1.score(X_train, y_train))
print(M2.score(X_train, y_train))
print(M3.score(X_train, y_train))

M1_pred = M1.predict(X_train)
M2_pred = M2.predict(X_train)
M3_pred = M3.predict(X_train)

train_prediction = {
 
    "DT" : M1_pred,
    "SVM" : M2_pred,
    "LR" : M3_pred
   
   
}
train_predictions = pd.DataFrame(train_prediction)
train_predictions

from sklearn.ensemble import RandomForestClassifier
smodel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

smodel.fit(train_predictions, y_train)

smodel.score(train_predictions, y_train)

"""## Predicting the Test set results"""

M1_pred_test = M1.predict(X_test)
M2_pred_test = M2.predict(X_test)
M3_pred_test = M3.predict(X_test)

test_prediction = {

    "DT" : M1_pred_test,
    "SVM" : M2_pred_test,
    "LR" : M3_pred_test, 

}
test_predictions = pd.DataFrame(test_prediction)
test_predictions

final_prediction = smodel.predict(test_predictions)

final_prediction

smodel.score(test_predictions, y_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, final_prediction)

print(accuracy_score(y_train, M1_pred))
print(accuracy_score(y_train, M2_pred))
print(accuracy_score(y_train, M3_pred))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, final_prediction)
print(cm)
print(accuracy_score(y_test, final_prediction))

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

cm = confusion_matrix(y_test, final_prediction)
print(cm)
print(accuracy_score(y_test, final_prediction))

accuracy = accuracy_score(y_test, final_prediction)
# accuracy: (tp + tn) / (p + n)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, final_prediction)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, final_prediction)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, final_prediction)
print('F1 score: %f' % f1)

# Saving our classifier
with open('C:/DEPLOYMENT/M1.pkl','wb') as m1:
    pickle.dump(M1,m1)
with open('C:/DEPLOYMENT/M2.pkl','wb') as m2:
    pickle.dump(M2,m2)
with open('C:/DEPLOYMENT/M3.pkl','wb') as m3:
    pickle.dump(M3,m3)
with open('C:/DEPLOYMENT/SMODEL.pkl','wb') as smod:
    pickle.dump(smodel,smod)    
    
# Saving the BAG OF WORDS model
with open('C:/DEPLOYMENT/Stack_bagofwordsmodel.pkl','wb') as Stack_BOW:
    pickle.dump(cv,Stack_BOW)
    
    