# Importing the libnraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the datasets
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

# Using elbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
WSCC=[]
for i in range (1,11):
    kmeans = KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X)
    WSCC.append(kmeans.inertia_)
plt.plot(range(1,11),WSCC)
plt.title('Elbow Graph')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show()
    
# Training K-Model on the datasets

kmeans = KMeans(n_clusters=5, random_state=42)
y_keams = kmeans.fit_predict(X)    

# Visualising the cluster 

plt.scatter(X[y_keams==0 , 0],X[y_keams==0 , 1],s=100,c='red', label='Cluster 1')
plt.scatter(X[y_keams==1 , 0],X[y_keams==1 , 1],s=100,c='blue', label='Cluster 2')
plt.scatter(X[y_keams==2 , 0],X[y_keams==2 , 1],s=100,c='green', label='Cluster 3')
plt.scatter(X[y_keams==3 , 0],X[y_keams==3 , 1],s=100,c='cyan', label='Cluster 4')
plt.scatter(X[y_keams==4 , 0],X[y_keams==4 , 1],s=100,c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Cluster of Customer')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()