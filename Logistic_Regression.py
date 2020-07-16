#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,0:2].values
y=dataset.iloc[:,2].values

#spiliting the data into training data and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#feature Scaling to improve the predictions 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#training the logistic regression on the model
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)

#predicting the new result
log.predict(sc.transform([[45,87000]]))

#predicting the test set results
y_pred=log.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

#accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)