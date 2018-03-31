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

# Taking care of missing data--------------------------
from sklearn.preprocessing import Imputer

# repaces the missing value with mean of the columnn :: axis=0 is for columnn.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#fit the imputer with the data objects, fitting only column 2 i.e index 1 and column 3 i.e index 2.
# X[:, 1:3] as uppper bound is excluded therefore 1:3 instead of 1:2
imputer = imputer.fit(X[:, 1:3])

# replace the missing data of the X
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data-------------------------------------------
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

# create a label for first column
# replaces category name to number
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Now as the column names are number, it should not think that one category is greater than other.
# therefore we will use onhotencoder to map it to dummy variable feature category.
#  adn changes from one column to 3 column as germany, spain and france.
onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set-----------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling----------------------------
# feature scaling is used to avoid the problem of large number and small number
#  because we are using euclidean distance i.e srqt( (x2-x1)**2 +(y2-y1)**2)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

# fit to perform the scaling
X_train = sc_X.fit_transform(X_train)

# we don't need to fit in here because it's already performed  in training data
#  point to keep in mind, about scaling the dummy variable : in these we are doing it.
X_test = sc_X.transform(X_test)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)