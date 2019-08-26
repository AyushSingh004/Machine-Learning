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
#Encoding categorial variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:, 0]=le.fit_transform(X[: , 0])
X

#Dumy Encoding using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray()

#for y
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
y= le1.fit_transform(y)





















