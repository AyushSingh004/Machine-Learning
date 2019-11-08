#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
dataset=pd.read_csv('50_Startups.csv')

X=dataset.iloc[ : , :-1 ].values
#X=pd.DataFrame(X)
y=dataset.iloc[ : , 4].values
#y=pd.DataFrame(y)

#Encoding categorial variable for X
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
X[:, 3]=labelencoder_X.fit_transform(X[: , 3])
# if this code doesn't work then type :-
#X.iloc[:,3]=labelencode_X.fit_transform(X[:,3])
X

#Dummy Encoding using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features= [3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable Trap
X = X[ : , 1 : ]

#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
    
#Predicting the Test set results
y_pred = lin_reg.predict(X_test)

#Testing the score of the variables
lin_reg.score(X_train,y_train)
lin_reg.score(X_test,y_test)
lin_reg.score(X,y)































