import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from math import sqrt

# read test and train csv files and store them in train and test variables
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

x_train = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]]
y_train = train["Survived"]

x_test = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]]

x_train.is_copy = False
x_test.is_copy = False

print(train.shape)
print(x_train.shape)
# Convert the male and female groups to integer form
maleRows = x_train["Sex"] == "male"
femaleRows = x_train["Sex"] == "female"
x_train.loc[maleRows, "Sex"] = 0
x_train.loc[femaleRows, "Sex"] = 1

# Impute the Embarked variable
x_train["Embarked"] = x_train["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
sRows = x_train["Embarked"] == "S"
cRows = x_train["Embarked"] == "C"
qRows = x_train["Embarked"] == "Q"
x_train.loc[sRows, "Embarked"] = 0
x_train.loc[cRows, "Embarked"] = 1
x_train.loc[qRows, "Embarked"] = 2

# Impute the Age variable
x_train["Age"] = x_train["Age"].fillna(x_train["Age"].median())

x_test.Fare[152] = x_test.Fare.median()
x_test["Age"] = x_test["Age"].fillna(x_test["Age"].median())
x_test["Pclass"] = x_test["Pclass"].fillna(x_test["Pclass"].median())
x_test["Fare"] = x_test["Fare"].fillna(x_test["Fare"].median())

maleRows = x_test["Sex"] == "male"
femaleRows = x_test["Sex"] == "female"
x_test.loc[maleRows, "Sex"] = 0
x_test.loc[femaleRows, "Sex"] = 1

x_test["Sex"] = x_test["Sex"].fillna(x_test["Sex"].median())

x_test["Embarked"] = x_test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
sRows = x_test["Embarked"] == "S"
cRows = x_test["Embarked"] == "C"
qRows = x_test["Embarked"] == "Q"

x_test.loc[sRows, "Embarked"] = 0
x_test.loc[cRows, "Embarked"] = 1
x_test.loc[qRows, "Embarked"] = 2

DT = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=1)

forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)

NN = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

LR = LogisticRegression(penalty='l1', tol=0.01)

eclf = VotingClassifier(estimators=[('forest', forest), ('DT', DT), ('LR',LR)], voting='hard')


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
    TPR = (TP*1.0) / (TP + FN)
    return TPR

X = x_train.values
y = y_train.values
kf = KFold(n_splits=10, random_state=None, shuffle=False)
DTRMSE = 0
NNRMSE = 0
ForestRMSE = 0
LRRMSE = 0
eclf_RMSE = 0
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
    #
    # NN.fit(X_train, y_train)
    # NNRMSE += sqrt(mean_squared_error(y_test, NN.predict(X_test)))
    # NN_TPR += perf_measure(y_test, NN.predict(X_test))
    #
    # forest.fit(X_train, y_train)
    # ForestRMSE += sqrt(mean_squared_error(y_test, forest.predict(X_test)))
    # forest_TPR += perf_measure(y_test, forest.predict(X_test))
    #
    #
    # LR.fit(X_train, y_train)
    # LRRMSE += sqrt(mean_squared_error(y_test, LR.predict(X_test)))
    # LR_TPR += perf_measure(y_test, LR.predict(X_test))

    eclf.fit(X_train, y_train)
    eclf_RMSE += sqrt(mean_squared_error(y_test, eclf.predict(X_test)))
    eclf_TPR += perf_measure(y_test, eclf.predict(X_test))


# print("DT: %f " % (DT_TPR / 10.0))
# print("DTRMSE: %f " % (DTRMSE / 10.0))
#
#
# print("NNTPR: %f " % (NN_TPR / 10.0))
# print("NNRMSE: %f " % (NNRMSE / 10.0))
#
#
# print("forest: %f " % (forest_TPR / 10.0))
# print("forestRMSE: %f " % (ForestRMSE / 10.0))
#
#
# print("LR_TPR: %f " % (LR_TPR / 10.0))
# print("LRRMSE: %f " % (LRRMSE / 10.0))


print("eclf_RMSE: %f " % (eclf_RMSE / 10.0))
print("eclf_TPR: %f " % (eclf_TPR / 10.0))


my_prediction = eclf.predict(x_test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])

# # Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution.csv", index_label=["PassengerId"])
