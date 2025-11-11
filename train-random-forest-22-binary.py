# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 11:22:46 2025

@author: user
"""

import pandas as pd
train_df = pd.read_csv("train-22-binary.csv")
print(train_df.head())

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
x = train_df[["E-disgust","E-anger","E-sad","S-very-neg","S-neg"]].values
y = train_df["V-sad"].values

rfc = RandomForestClassifier()

rfc.fit(x,y)

import numpy as np

#row: [EMOTION, SENTIMENT]
new_data = np.array([
    [0,0,0,0,0],
    [0,0,0,0,1],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [1,0,0,1,0]])

y_pred = rfc.predict(new_data)

print("__________________________")
print("Predictions: {}".format(y_pred))


from sklearn.model_selection import train_test_split

x = train_df.drop("V-sad", axis=1).values
y = train_df["V-sad"].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=21, stratify=y)

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

print(rfc.score(x_test, y_test))


