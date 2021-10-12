import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

dataset = load_iris()

X = dataset.data
y = dataset.target
#this graph represent only the bases of sepal length or width 
plt.scatter(X[y == 0,0],X[y == 0,1],c="r",label="setosa")
plt.scatter(X[y == 1,0],X[y == 1,1],c="g",label="versicolor")
plt.scatter(X[y == 2,0],X[y == 2,1],c="b",label="verginica")
plt.title("Analysis of Iris Dataset")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()
plt.show()


#this graph represent only the bases of sepal length or petal length 
plt.scatter(X[y == 0,0],X[y == 0,2],c="r",label="setosa")
plt.scatter(X[y == 1,0],X[y == 1,2],c="g",label="versicolor")
plt.scatter(X[y == 2,0],X[y == 2,2],c="b",label="verginica")
plt.title("Analysis of Iris Dataset")
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend()
plt.show()


#this graph represent only the bases of sepal length or petal width 
plt.scatter(X[y == 0,0],X[y == 0,3],c="r",label="setosa")
plt.scatter(X[y == 1,0],X[y == 1,3],c="g",label="versicolor")
plt.scatter(X[y == 2,0],X[y == 2,3],c="b",label="verginica")
plt.title("Analysis of Iris Dataset")
plt.xlabel("sepal length")
plt.ylabel("petal width")
plt.legend()
plt.show()


#this graph represent only the bases of sepal width or petal length
plt.scatter(X[y == 0,1],X[y == 0,2],c="r",label="setosa")
plt.scatter(X[y == 1,1],X[y == 1,2],c="g",label="versicolor")
plt.scatter(X[y == 2,1],X[y == 2,2],c="b",label="verginica")
plt.title("Analysis of Iris Dataset")
plt.xlabel("sepal width")
plt.ylabel("petal length")
plt.legend()
plt.show()


#this graph represent only the bases of sepal width   or petal width
plt.scatter(X[y == 0,1],X[y == 0,3],c="r",label="setosa")
plt.scatter(X[y == 1,1],X[y == 1,3],c="g",label="versicolor")
plt.scatter(X[y == 2,1],X[y == 2,3],c="b",label="verginica")
plt.title("Analysis of Iris Dataset")
plt.xlabel("sepal width")
plt.ylabel("petal width")
plt.legend()
plt.show()


#this graph represent only the bases of petal length or petal width 
plt.scatter(X[y == 0,2],X[y == 0,3],c="r",label="setosa")
plt.scatter(X[y == 1,2],X[y == 1,3],c="g",label="versicolor")
plt.scatter(X[y == 2,2],X[y == 2,3],c="b",label="verginica")
plt.title("Analysis of Iris Dataset")
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend()
plt.show()