#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Salary_Data.csv')

#importing the data
X=dataset.iloc[:,0].values
y=dataset.iloc[:,-1].values

#making feature matrix
X=np.c_[X,np.ones(30)]


#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#fitting simple linear regression into the training set
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_train,y_train)

#to determine the score of our dataset
lin.score(X_train,y_train)
lin.score(X_test,y_test)
lin.score(X,y)

#predicting the test set result
y_pred= lin.predict(X_test)

#making the co-ordinate of equal size
y=np.c_[y,np.ones(30)]

#Visualising of training set result
plt.scatter(X_train,y_train,color="blue")
plt.plot(X_train,lin.predict(X_train),color="red")
plt.title('SALARY vs EXPERIENCE (TRAINING SET)')
plt.xlabel('AGE')
plt.ylabel('SALARY')
plt.show()

plt.scatter(X_test,y_test,color="blue")
plt.plot(X_test,lin.predict(X_test),color="red")
plt.title('SALARY vs EXPERIENCE (TRAINING SET)')
plt.xlabel('AGE')
plt.ylabel('SALARY')
plt.show()

plt.scatter(X,y,color="blue")
plt.plot(X,lin.predict(X),color="red")
plt.title('SALARY vs EXPERIENCE (TRAINING SET)')
plt.xlabel('AGE')
plt.ylabel('SALARY')
plt.show()
