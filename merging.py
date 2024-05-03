import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

injury_df = pd.read_csv('InjuryRecord.csv')
playlist_df = pd.read_csv('PlayList.csv')

# Perform a left merge to keep all rows from playlist_df and add injury details where available
merged_df = pd.merge(playlist_df, injury_df, on=['PlayerKey', 'GameID', 'PlayKey'], how='left')

# Reorder the columns if needed
reordered_columns = ['PlayerKey', 'GameID', 'PlayKey', 'FieldType', 'PlayType', 'PlayerDay', 'PlayerGame', 'PlayerGamePlay', 'Position', 'PositionGroup', 'RosterPosition', 'StadiumType', 'Temperature', 'Weather', 'BodyPart', 'DM_M1', 'DM_M7', 'DM_M28', 'DM_M42']
merged_df = merged_df[reordered_columns]

# Remove '-' symbol from GameID and PlayKey columns
merged_df['GameID'] = merged_df['GameID'].str.replace('-', '')
merged_df['PlayKey'] = merged_df['PlayKey'].str.replace('-', '')

# Create the 'Days_Out' column and drop individual 'DM' columns
def calculate_days_out(row):
    if row['DM_M42'] == 1:
        return 42
    elif row['DM_M28'] == 1:
        return 28
    elif row['DM_M7'] == 1:
        return 7
    elif row['DM_M1'] == 1:
        return 1
    else:
        return 0  # If no '1' is found

# Apply the function along the rows to create the 'Days_Out' column
merged_df['Days_Out'] = merged_df.apply(calculate_days_out, axis=1)

# Drop the individual 'DM' columns
merged_df = merged_df.drop(['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42'], axis=1)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('sample2.csv', index=False)
