#Polynominal Regression

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[ : , 1:2].values
#X=pd.DataFrame(X)
y=dataset.iloc[ : , 2].values
#y=pd.DataFrame(y)

#Fitting Linear Regression to the datasets
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynominal Regression to the datasets
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures( degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the Linear Regression result
plt.scatter(X,y, color ='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('True or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynominal Regression result
X_grid = np.arange(min(X),max(X),0.1)
X_grid =X_grid.reshape(len(X_grid),1)
plt.scatter(X,y, color ='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('True or Bluff (Polynominal Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting the new result in Linear regression
X_pred=lin_reg.predict([[6.5]])

#Predicting the new result in Polynomial regression
X_pred_2=lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))