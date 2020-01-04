# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:05:38 2019

@author: Padmini
"""

import numpy as np #for linear algebra
import pandas as pd #to read csv and xlsx files, here we have xlsx file for our project - ALF_Data.xlsx

df = pd.read_excel ('C:\padmini\FALL 2019\ML CSC 5825\ML TERM PROJECT STUFF\ALF_Data.xlsx')
print (df)

#importing functions for preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix

#importing functions for visualizations
import seaborn as sns
from matplotlib import pyplot as plt

#Exploratory data analysis
copy_df=df
df.head() #8785 rows*30 columns
df.shape
df.info()

#some preprocessing implementations to make the large file easier to handle
#dropping samples that don't have ALF value
df=df.dropna(axis=0,subset=['ALF']) #colums is reduced to 30 after this
df.shape
df.isnull()

#after removing the null values, adding up the total missing values
missingvalues=df.isnull().sum()
missingvalues

#calculating correlation matrix
corr=df.corr()
#plotting a map for this - just for visualization purposes
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)#heatmap produced
df.corr()

#preparing for train-test split
y=df['ALF']
df=df.drop('ALF',axis=1)
df.head()
X=df
#splitting into train, test and validation
X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,shuffle=True ,test_size=0.2)
X_train.head()
y_train.head()

#impmeneting in SVM - Support Vector Machine
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#making predictions
y_pred = svclassifier.predict(X_test)

#For evaluating the algorithm - confusion matrix, precision and recall are used
from sklearn.metrics import classification_report, confusion_matrix #metrics library from scikit contains confusion matrix methods which can be used here
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))