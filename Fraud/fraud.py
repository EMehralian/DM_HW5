import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# read test and train csv files and store them in train and test variables
x_train = pd.read_csv('./data_fraud/X_train.csv')
y_train = pd.read_csv('./data_fraud/Y_train.csv')
# print(x_train.head())


# print(x_train.describe())

# drop two features from dataframe
x_train = x_train.drop("total", axis=1)
x_train = x_train.drop("state", axis=1)
print(x_train.shape)

# number of different account numbers
print(len(set(x_train["customerAttr_a"])))

# number of different email addresses
print(len(set(x_train["customerAttr_b"])))

# concat x_train and y_train
x_train = pd.concat([x_train, y_train], axis=1)

# fraud detected rows in dataframe (2654 rows)
fraud = x_train["fraud"] == 1
fraud_rows = x_train[fraud]

# repeat each fraud tuple 5 more times in the data frame
x_train = x_train.append([fraud_rows] * 5, ignore_index=True)
# resulted dataframe has 113270 rows which 15924 rows are fraud
print(x_train.shape)
