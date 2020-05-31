# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
dtr= DecisionTreeRegressor(random_state=0)
dtr.fit(X,y)

# Predicting a new result
dtr.predict([[4.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid)),1)
plt.scatter(X,y,color='r')
plt.plot(X_grid,dtr.predict(X_grid),color='b')
plt.show()
