# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:35:21 2019

@author: Karthick Ragavendran
"""

# Importing files
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import io

updates_folder='data/'
updates_folder_files=os.listdir(updates_folder)
data_hub_filename="data_hub.csv"

#Making it one file

def watch_new_data():
    print("Entering Watch Data")
    df_hub=pd.read_csv(updates_folder + data_hub_filename)
    print("Before all merging:", df_hub.shape)

    if(len(updates_folder_files)>1):
        updates_folder_files.remove("data_hub.csv")
        print("Number of new files received:", len(updates_folder_files))
        for i in updates_folder_files[0:]:
            print("file Name:", i)
            df_temp= pd.read_csv(updates_folder + i)
            print("Merging file {filenum} in the hub".format(filenum=i))
            df_hub=df_hub.append(df_temp)
            print("Now the data size is:",df_hub.shape)
            df_hub.to_csv(updates_folder + data_hub_filename)
            os.remove('data/' + i)
        return(1)
    else:
        return(0)
    


    
#Train Model
def train_model(df):

    #Preprocessing
    LE=LabelEncoder()
    categorical_cols=['Neighbourhood', 'No-show','Gender']
    df[categorical_cols] = df[categorical_cols].apply(lambda col: LE.fit_transform(col.astype(str)))

    #Logistic Regression
    log=LogisticRegression()
    features=['Age','Hipertension','Diabetes','SMS_received']
    X=df[features]
    Y=df["No-show"]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.15)

    log.fit(x_train,y_train)
    y_pred=log.predict(x_test)
    pickle.dump(log,open("show_pred_model_2.pkl","wb"))

    accuracy_score1 = accuracy_score(y_test,y_pred)
    print(accuracy_score1)
    

if(watch_new_data()==1):
    print("New Files Added")
    df=pd.read_csv(updates_folder + data_hub_filename)
    train_model(df)
else:
    print("The model is up to date...")