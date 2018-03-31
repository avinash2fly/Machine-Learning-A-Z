# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer

# repaces the missing value with mean of the columnn :: axis=0 is for columnn.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#fit the imputer with the data objects, fitting only column 2 i.e index 1 and column 3 i.e index 2.
# X[:, 1:3] as uppper bound is excluded therefore 1:3 instead of 1:2
imputer = imputer.fit(X[:, 1:3])

# replace the missing data of the X
X[:, 1:3] = imputer.transform(X[:, 1:3])