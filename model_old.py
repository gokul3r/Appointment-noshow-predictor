# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:11:36 2019

@author: Karthick Ragavendran
"""

# Importing files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

#Import Data
df=pd.read_csv("KaggleV2-May-2016.csv")
df.head()


#Preprocessing
LE=LabelEncoder()
categorical_cols=['Neighbourhood', 'No-show','Gender']
df[categorical_cols] = df[categorical_cols].apply(lambda col: LE.fit_transform(col.astype(str)))
df.info()


#Get Scheduled Weekday, Appointed WEekday
import datetime  
from datetime import date 
import calendar 
def findDay(date): 
    year, month, day = (int(i) for i in date.split('-')) 
    born = datetime.date(year, month, day) 
    return born.strftime("%A") 

#Logistic Regression
log=LogisticRegression()
features=['Age','Hipertension','Diabetes','SMS_received']
X=df[features]
Y=df["No-show"]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.15)
print(x_train.shape)
print(x_test.shape)

log.fit(x_train,y_train)
y_pred=log.predict(x_test)
pickle.dump(log,open("show_pred_model_1.pkl","wb"))

accuracy_score = accuracy_score(y_test,y_pred)
print(accuracy_score)