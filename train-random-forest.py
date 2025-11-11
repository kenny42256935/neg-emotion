# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 11:22:46 2025

@author: user
"""

import pandas as pd
train_df = pd.read_csv("train.csv")
print(train_df.head())

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
x = train_df[["EMOTION","SENTIMENT"]].values
y = train_df["VISION"].values

rfc = RandomForestClassifier()

rfc.fit(x,y)

import numpy as np

#row: [EMOTION, SENTIMENT]
new_data = np.array([
    [8,5],
    [7,5],
    [6,5],
    [8,4],
    [7,4],
    [6,4],
    [8,3],
    [7,3],
    [6,3]])

y_pred = rfc.predict(new_data)

print("__________________________")
print("Predictions: {}".format(y_pred))


from sklearn.model_selection import train_test_split

x = train_df.drop("VISION", axis=1).values
y = train_df["VISION"].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=21, stratify=y)

rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

print(rfc.score(x_test, y_test))


