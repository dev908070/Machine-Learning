
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

##step 1 :- get the data
dataset = pd.read_excel("blood.xlsx")

#seperate the Feture matrix or vector of prediction

X = dataset.iloc[2:,1].values
X = X.reshape(-1,1)
y = dataset.iloc[2:,-1].values

## step 2 :- Discovery and Data visualization
plt.scatter(X,y)
plt.xlabel("age")
plt.ylabel("Systolic blood pressure")
plt.title("Analysis for bp ")
plt.show()

## step 3 :- Data Preprocessing 


## step 4 :- Applying the Machine Learning  Model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(X,y)

lin_reg.predict([[25]])
lin_reg.predict([[30]])
lin_reg.predict([[24]])


"""Logic behind this Algo :-    Simple linear Regression must have found 
                                a line of prediction which it is use to generate 
                                prediction if it is a line , it will certainlay 
                                have a gradient and intercept"""

### formula of line is here 

Beta_1 = lin_reg.coef_
Beta_1

Beta_0 = lin_reg.intercept_
Beta_0



#if we want to calculate or predict age of a group

y_pred = lin_reg.predict(X)




plt.scatter(X,y)
plt.plot(X,y_pred, c= "r")
plt.xlabel("age")
plt.ylabel("Systolic blood pressure")
plt.title("Analysis for bp ")
plt.show()


lin_reg.score(X,y)

"""we have to varify in Data preprocessing there is no outlier present in the 
   dataset  if we want more accuracey we have to remove outliers
   after that we have to run score() method to verify of the model accuracey"""
outlier present in the dataset
 



