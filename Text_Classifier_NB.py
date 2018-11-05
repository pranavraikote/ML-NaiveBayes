# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5 19:05:38 2018

@author: Pranav
"""
#Import Modules
import pandas as pd

#Import the data from CSV 
msg=pd.read_csv('Classify_text_data.csv',names=['message','label'])
print('The dimensions of the dataset',msg.shape)


#Categorize the data with labels
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.labelnum
print(X)
print(Y)


#Splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y)


#Optional, to see the shape of Training and Testing Data
print(xtest.shape)
print(xtrain.shape)
print(ytest.shape)
print(ytrain.shape)


#Output of count vectoriser is a sparse matrix, convert bag of words to matrix form
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
print(count_vect.get_feature_names())


#Tabular representation
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df)


# Training Multinomial Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)

#Printing accuracy of the model, confusion matrix, recall and precision
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))