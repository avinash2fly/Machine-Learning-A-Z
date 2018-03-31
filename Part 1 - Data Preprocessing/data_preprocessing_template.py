# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset-------------------
dataset = pd.read_csv('Data.csv')

# take all the lines and except the last columns.
X = dataset.iloc[:, :-1].values
# take all the lines/rows and only the last columns.
y = dataset.iloc[:, 3].values



# Splitting the dataset into the Training set and Test set-----------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling----------------------------
# feature scaling is used to avoid the problem of large number and small number
#  because we are using euclidean distance i.e srqt( (x2-x1)**2 +(y2-y1)**2)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# fit to perform the scaling
X_train = sc_X.fit_transform(X_train)

# we don't need to fit in here because it's already performed  in training data
#  point to keep in mind, about scaling the dummy variable : in these we are doing it.
X_test = sc_X.transform(X_test)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""