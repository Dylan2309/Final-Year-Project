import unittest
from unittest.mock import patch
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from RFkfold import read_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier

class TestModel(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_read_data(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'PlayKey_y': [1, 2],
            'PlayType': ['Pass', 'Run'],  # Correct as per your data
            'FieldType': ['Turf', 'Grass'],
            'Position': ['QB', 'RB'],
            'Temperature': [70, 75],
            'Weather_Category': ['Clear', 'Rainy'],
            'WorkloadIncrease': [5, 10],
            'PreInjuryWorkload': [15, 20],
            'GameNumberInSeason': [1, 2],
            'Injured': [0, 1]
        })
        
        X, y = read_data()
        expected_X = mock_read_csv.return_value.drop(columns=['Injured'])
        expected_y = mock_read_csv.return_value['Injured']
        assert_frame_equal(X, expected_X)
        assert_series_equal(y, expected_y)

    @patch('pandas.read_csv')
    def test_preprocess_data(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'PlayKey_y': [1, 2, 3],
            'PlayType': ['Pass', 'Run', 'Pass'],
            'FieldType': ['Turf', 'Grass', 'Turf'],
            'Position': ['QB', 'RB', 'QB'],
            'Temperature': [70, 75, 80],
            'Weather_Category': ['Clear', 'Rainy', 'Sunny'],
            'WorkloadIncrease': [5, 10, 5],
            'PreInjuryWorkload': [15, 20, 15],
            'GameNumberInSeason': [1, 2, 1],
            'Injured': [0, 1, 0]
        })
        
        X, y = read_data()
        X_preprocessed, y_preprocessed = preprocess_data(X, y)
        
        self.assertIn('PlayType_Pass', X_preprocessed.columns)
        self.assertIn('PlayType_Run', X_preprocessed.columns)
        self.assertIn('FieldType_Grass', X_preprocessed.columns)
        self.assertIn('FieldType_Turf', X_preprocessed.columns)
        self.assertIn('Position_QB', X_preprocessed.columns)
        assert_series_equal(y, y_preprocessed)


if __name__ == '__main__':
    unittest.main()
