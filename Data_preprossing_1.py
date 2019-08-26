import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')

X=dataset.iloc[ : , 0:3 ].values
#X=pd.DataFrame(X)
y=dataset.iloc[ : , -1].values
#y=pd.DataFrame(y)

from sklearn.impute import SimpleImputer
sim=SimpleImputer()
sim.fit(X[:,1:3])
X[:, 1:3]=sim.transform(X[:,1:3])
X











