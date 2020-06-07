#Random Forest Regression

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Random Tree Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X,y)

# Predicting a new result
regressor.predict([[4.5]])

# Visualising the Random Tree Regression results (higher resolution)
X_grid = np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='r')
plt.plot(X_grid,regressor.predict(X_grid),color='b')
plt.show()














