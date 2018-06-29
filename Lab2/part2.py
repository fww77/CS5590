import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.naive_bayes import GaussianNB as Guass
from sklearn.neighbors import KNeighborsClassifier
import pandas

# Load the Data Set
data_set = datasets.load_wine()
# Assign X and y variables
X = data_set.data
y = data_set.target
# Split the data 60/40 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# Create the model
model = Guass()
# Fit the test data
model.fit(X_test,y_test)
# Print the outcome
print(cross_val_score(model, X_test, y_test, cv=10))