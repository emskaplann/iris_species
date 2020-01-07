import sys
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import scipy as sp
import sklearn
from sklearn.model_selection  import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

iris_dataset = load_iris()
# print(iris_dataset)
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
# print(X_train.shape, y_train.shape)
# putting our data into k nearest neighbors algo
knn.fit(X_train, y_train)
# this is for testing a single iris - optional
X_new=np.array([[5, 3.1, 1, 4.2]])
# testing the accuracy
prediction = knn.score(X_test, y_test)
print("accuracy rate:", prediction)
