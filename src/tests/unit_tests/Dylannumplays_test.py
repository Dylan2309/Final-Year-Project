import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

# from data_processing import categorize_stadium, weather_mapping

def categorize_stadium(stadium_type):
    """Categorizes StadiumType into 'Indoor' or 'Outdoor'."""
    indoor_types = ['dome', 'closed dome', 'dome, closed', 'domed, closed', 
                    'retr. roof-closed', 'retr. roof - closed', 'retr. roof closed', 
                    'indoor', 'indoors', 'indoor, roof closed', 'domed']
    if str(stadium_type).lower() in indoor_types:
        return 'Indoor'
    else:
        return 'Outdoor'

weather_mapping = {
    'Clear and warm': 'dry',
    'Rain': 'wet',
    'Snow': 'snowy',
    'Sunny': 'dry',
    'Indoor': 'indoor',
    'nan': 'Other'
}

class TestDataProcessing(unittest.TestCase):

    def test_categorize_stadium(self):
        # Test categorizing stadium types
        self.assertEqual(categorize_stadium("dome"), "Indoor")
        self.assertEqual(categorize_stadium("closed dome"), "Indoor")
        self.assertEqual(categorize_stadium("open stadium"), "Outdoor")

    def test_weather_category_mapping(self):
        # Prepare data
        test_data = pd.DataFrame({
            'Weather': ['Clear and warm', 'Rain', 'Snow', 'Sunny', 'Indoor', 'nan']
        })
        expected_results = pd.DataFrame({
            'Weather': ['Clear and warm', 'Rain', 'Snow', 'Sunny', 'Indoor', 'nan'],
            'Weather_Category': ['dry', 'wet', 'snowy', 'dry', 'indoor', 'Other']
        })
        
        # Apply the mapping
        test_data['Weather_Category'] = test_data['Weather'].map(weather_mapping)
        
        # Test whether actual results match expected results
        assert_frame_equal(test_data, expected_results)

    def test_apply_stadium_categorization(self):
        # Prepare data
        test_df = pd.DataFrame({
            'StadiumType': ['dome', 'open stadium', 'closed dome', 'indoor']
        })
        expected_df = pd.DataFrame({
            'StadiumType': ['dome', 'open stadium', 'closed dome', 'indoor'],
            'StadiumCategory': ['Indoor', 'Outdoor', 'Indoor', 'Indoor']
        })
        
        # Apply the function
        test_df['StadiumCategory'] = test_df['StadiumType'].apply(categorize_stadium)
        
        # Check results
        assert_frame_equal(test_df, expected_df)

if __name__ == '__main__':
    unittest.main()
