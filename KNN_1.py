# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:43:51 2024

@author: Admin
"""

import pandas as pd
import numpy as np
glass=pd.read_csv("C:/naive_bayes_theorem/glass.csv.xls")
glass.describe()
glass.info()

#Normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this function to the dataframe
glass_n=norm_func(glass.iloc[:,1:10])

#Let us apply X as input and y as oputput
X=np.array(glass_n.iloc[:,:])
#wbcd_n are already excluding output column 
y=np.array(glass['Type'])

#Let us split the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

"""
Here you are passing X,y instead dataframe handle there could 
be chances of unbalancing data let us assume that we have 100 
points out of which 80 nc and 10 cancer these data ponts must 
be equally distributed there is satisfied sampling concept used
"""
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
#Let us evaluate the model
#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))
pd.crosstab(pred,y_test)

#Let us try to select the correct value of k
acc=[]
#Running knn algorithm for k=3 t0 50 in step of 2
#K value selected is oddd value
for i in range(3,30,2):
    #Declare the model
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc=np.mean(neigh.predict(X_train)==y_train)
    test_acc=np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])
#if you will see the acc it has got teo accuracy, 
#i[0]-train_acc
#to plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,30,2),[i[0]for i in acc],"ro-") 
plt.plot(np.arange(3,30,2),[i[0]for i in acc],"bo-")
#There are 3,5,7 and 9 are possible values where 
#accuracy is good let us check for k-3
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
pred
accuracy_score(pred,y_test)
pd.crosstab(pred, y_test)













   