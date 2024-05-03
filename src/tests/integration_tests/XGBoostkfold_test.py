import os
import pytest
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np  
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from XGBoostkfold import read_data, preprocess_data

def test_model_training_runs():
    X_filepath = 'src/tests/Xtest_data.csv'
    y_filepath = 'src/tests/ytest_data.csv'  
    X, y = read_data(X_filepath, y_filepath)
    X, y = preprocess_data(X, y)
    model = xgb.XGBClassifier()
    model.fit(X, y)  
    print("passed training runs")

def test_cross_validation_runs():
    X_filepath = 'src/tests/Xtest_data.csv'
    y_filepath = 'src/tests/ytest_data.csv'  
    X, y = read_data(X_filepath, y_filepath)
    X, y = preprocess_data(X, y)
    model = xgb.XGBClassifier()
    skf = StratifiedKFold(n_splits=2)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        print("passed cross validation test")

def test_cross_validation():
    X = np.array([[1, 2], [3, 4], [5, 6], [2, 6], [7, 9]])
    y = pd.read_csv('src/tests/ytest_data.csv')['Injured']

    model = xgb.XGBClassifier()
    skf = StratifiedKFold(n_splits=2)

    exception_raised = False
    try:
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
    except Exception as e:
        exception_raised = True
        print(f"Exception raised during cross-validation: {str(e)}")

    assert not exception_raised, "Cross-validation should not raise any exceptions."
