#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Data.csv')

X=dataset.iloc[ : , 0:3 ].values
#X=pd.DataFrame(X)
y=dataset.iloc[ : , -1].values
#y=pd.DataFrame(y)

#Taking care of missing data
from sklearn.impute import SimpleImputer
sim=SimpleImputer()
sim.fit(X[:,1:3])
X[:, 1:3]=sim.transform(X[:,1:3])
X
#Encoding categorial variable for X
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[: , 0])
X

#Dumy Encoding using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray()

#Encoding categorial variable for y
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#Feature Scaling to make the variable in the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



