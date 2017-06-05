import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error

# read test and train csv files and store them in train and test variables
x_train = pd.read_csv('./data_fraud/X_train.csv')
y_train = pd.read_csv('./data_fraud/Y_train.csv')
x_test = pd.read_csv('./data_fraud/X_test.csv')
# print(x_train.head())

# print(x_train.describe())

# drop two features from dataframe
x_train = x_train.drop("total", axis=1)
x_train = x_train.drop("state", axis=1)
x_train = x_train.drop("customerAttr_b", axis=1)

x_test = x_test.drop("total", axis=1)
x_test = x_test.drop("state", axis=1)
x_test = x_test.drop("customerAttr_b", axis=1)
# x_train_withoutEmail = x_train.drop("customerAttr_b", axis=1)
print(x_train.shape)

# number of different account numbers
print(len(set(x_train["customerAttr_a"])))

# number of different email addresses
# print(len(set(x_train["customerAttr_b"])))

# concat x_train and y_train
x_train = pd.concat([x_train, y_train], axis=1)

# fraud detected rows in dataframe (2654 rows)
fraud = x_train["fraud"] == 1
fraud_rows = x_train[fraud]

# repeat each fraud tuple 5 more times in the data frame
x_train = x_train.append([fraud_rows] * 5, ignore_index=True)
# resulted dataframe has 113270 rows which 15924 rows are fraud
print(x_train.shape)

y_train = x_train["fraud"]
x_train = x_train.drop("fraud", axis=1)


def remove_low_variance_features(data_set, threshold):
    sel = VarianceThreshold(threshold=threshold)
    return sel.fit_transform(data_set)


threshold = 0.8 * (1 - 0.8)  # Suppose data follows Bernoulli Distribution, thus variance = p(1-p)
new_x_train = remove_low_variance_features(x_train, threshold)
print(new_x_train.shape)

# Building and fitting forest
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
# my_forest = forest.fit(x_train, y_train)
# # Print the score of the fitted random forest
# print(my_forest.score(x_train, y_train))
# pred_forest = my_forest.predict(x_test)
# print(len(set(pred_forest == 1)))
# print(my_forest.feature_importances_)

DT = DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_leaf=5)
# my_DT = DT.fit(x_train, y_train)
# print(my_DT.score(x_train, y_train))
# pred_DT = my_DT.predict(x_test)
# print(len(set(pred_DT == 1)))

NN = MLPClassifier( alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

LR = LogisticRegression(penalty='l1', tol=0.01)

eclf = VotingClassifier(estimators=[('forest', forest), ('DT', DT), ('NN', NN)], voting='hard')

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    TPR = TP / (TP + FN)
    return TPR


X = x_train.values
y = y_train.values
kf = KFold(n_splits=10, random_state=None, shuffle=False)
DTRMSE = 0
NNRMSE = 0
ForestRMSE = 0
LRRMSE = 0
DT_TPR = 0
NN_TPR = 0
forest_TPR = 0
LR_TPR = 0
eclf_TPR = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # DT.fit(X_train, y_train)
    # DTRMSE += sqrt(mean_squared_error(y_test, DT.predict(X_test)))
    # DT_TPR += perf_measure(y_test, DT.predict(X_test))

    # NN.fit(X_train, y_train)
    # NNRMSE += sqrt(mean_squared_error(y_test, NN.predict(X_test)))
    # NN_TPR += perf_measure(y_test, NN.predict(X_test))

    # forest.fit(X_train, y_train)
    # ForestRMSE += sqrt(mean_squared_error(y_test, forest.predict(X_test)))
    # forest_TPR += perf_measure(y_test, forest.predict(X_test))

    # LR.fit(X_train, y_train)
    # LRRMSE += sqrt(mean_squared_error(y_test, LR.predict(X_test)))
    # LR_TPR += perf_measure(y_test, LR.predict(X_test))

    eclf.fit(X_train,y_train)
    eclf_TPR += perf_measure(y_test, eclf.predict(X_test))

# print("DT: %f " % (DT_TPR / 10))
# print("DTRMSE: %f " % (DTRMSE / 10))
# print("NN: %f " % (NN_TPR / 10))
# print("NNRMSE: %f " % (NNRMSE / 10))
# print("forest: %f " % (forest_TPR / 10))
# print("forestRMSE: %f " % (ForestRMSE / 10))
# print("LR_TPR: %f " % (LR_TPR / 10))
# print("LRRMSE: %f " % (LRRMSE / 10))
print("eclf_TPR: %f " % (eclf_TPR / 10))


my_prediction = eclf.predict(x_test)

df = pd.DataFrame({"fraud": my_prediction})
df.to_csv("my_prediction.csv", index=False)