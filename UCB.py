#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#implementation of UCB
import math
N=10000
d=10
ads_selected=[]
number_of_selection=[0]*d
sum_of_rewards=[0]*d
total_reward=0
for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        if (number_of_selection[i]>0):
            #average of ads
            average_of_ad=(sum_of_rewards[i]/number_of_selection[i])
            #confidence interval
            delta_i=math.sqrt(3/2*math.log(n+1)/number_of_selection[i])
            upper_bound=average_of_ad+delta_i
        else:
            upper_bound= 1e400 #(infinity)
            if (upper_bound>max_upper_bound):
                max_upper_bound=upper_bound
                ad=i
ads_selected.append(ad)
number_of_selection[ad]+=1
sum_of_rewards[ad]=sum_of_rewards[ad]+dataset.values[n,ad]#-->reward
total_reward=total_reward+dataset.values[n,ad]

#visualising the UCB
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('ads')
plt.ylabel('number of times each ads was selected')
plt.show()
