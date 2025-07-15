# ML
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Linear Classifier: Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log)}')
# Non-linear Classifier: Support Vector Machine (SVM)
svm = SVC(kernel='rbf') # RBF kernel for non-linearity
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print(f'SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}')
# Simple Linear Regression (on one feature, using Petal Length for
simplicity)
X_simple = X[:, 2].reshape(-1, 1) # Petal Length
X_train_simple, X_test_simple, y_train_simple, y_test_simple =
train_test_split(X_simple, y, test_size=0.2, random_state=42)
lin_reg_simple = LinearRegression()
lin_reg_simple.fit(X_train_simple, y_train_simple)
y_pred_simple = lin_reg_simple.predict(X_test_simple)
print(f'Simple Linear Regression MSE: {mean_squared_error(y_test_simple,
y_pred_simple)}')
# Multiple Linear Regression (using all features)
lin_reg_multiple = LinearRegression()
lin_reg_multiple.fit(X_train, y_train)
y_pred_multiple = lin_reg_multiple.predict(X_test)
print(f'Multiple Linear Regression MSE: {mean_squared_error(y_test,
y_pred_multiple)}')
