import os
import pytest
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from RFkfold import read_data, preprocess_data 

def test_file_loading():
    X_filepath = 'src/tests/Xtest_data.csv'
    y_filepath = 'src/tests/ytest_data.csv'  
    X, y = read_data(X_filepath, y_filepath)
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    print("passed DataFrame check")

def test_required_columns():
    X_filepath = 'src/tests/Xtest_data.csv'
    y_filepath = 'src/tests/ytest_data.csv'
    X, _ = read_data(X_filepath, y_filepath)
    required_columns = ['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease']
    assert all(col in X.columns for col in required_columns), "All required columns should be present in the DataFrame"
    print("passed required columns")

def test_training():
    X = pd.DataFrame({
        'WorkloadDensity': [1, 2, 3, 4, 5],
        'Temperature': [70, 75, 80, 85, 90]
    })
    y = pd.read_csv('src/tests/ytest_data.csv')['Injured']
    
    model = RandomForestClassifier()
    try:
        model.fit(X, y)
        exception_raised = False
    except Exception as e:
        exception_raised = True
        print(f"Exception raised during training: {str(e)}")

    assert not exception_raised, "Model training should not raise any exceptions."
