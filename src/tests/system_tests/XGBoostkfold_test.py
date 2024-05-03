import os
import pytest
import xgboost as xgb
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from XGBoostkfold import read_data, preprocess_data

def test_full_pipeline():
    # Load data
    X, y = read_data('src/tests/Xtest_data.csv', 'src/tests/ytest_data.csv')

    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)

    # Train model
    model = xgb.XGBClassifier()
    model.fit(X_processed, y_processed)

    # Make predictions
    predictions = model.predict(X_processed)

    # Evaluate predictions
    assert predictions is not None, "No predictions were made"
    assert len(predictions) == len(y_processed), "Number of predictions does not match number of samples"
