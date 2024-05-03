import pandas as pd
import numpy as np


# Read the merged data
df = pd.read_csv('MergedData_LeftMerge.csv')

# Feature Engineering for StadiumType
def categorize_stadium(stadium_type):
    """Categorizes StadiumType into 'Indoor' or 'Outdoor'."""
    indoor_types = ['dome', 'closed dome', 'dome, closed', 'domed, closed', 
                    'retr. roof-closed', 'retr. roof - closed', 'retr. roof closed', 
                    'indoor', 'indoors', 'indoor, roof closed', 'domed']
    if str(stadium_type).lower() in indoor_types:
        return 'Indoor'
    else:
        return 'Outdoor'

weather_column = 'Weather'  # Adjust this according to your dataset    

weather_mapping = {
    'Clear and warm': 'dry',
    'Mostly Cloudy': 'damp',
    'Sunny': 'dry',
    'Clear': 'dry',
    'Cloudy': 'damp',
    'Cloudy, fog started developing in 2nd quarter': 'damp',
    'Rain': 'wet',
    'Partly Cloudy': 'damp',
    'Mostly cloudy': 'damp',
    'Cloudy and cold': 'damp',
    'Cloudy and Cool': 'damp',
    'Rain Chance 40%': 'wet',
    'Controlled Climate': 'indoor',
    'Sunny and warm': 'dry',
    'Partly cloudy': 'damp',
    'Clear and Cool': 'dry',
    'Clear and cold': 'dry',
    'Sunny and cold': 'dry',
    'Indoor': 'indoor',
    'nan': 'Other',
    'Partly Sunny': 'damp',
    'N/A (Indoors)': 'indoor',
    'Mostly Sunny': 'dry',
    'Indoors': 'indoor',
    'Clear Skies': 'dry',
    'Partly sunny': 'dry',
    'Showers': 'wet',
    'N/A Indoor': 'indoor',
    'Sunny and clear': 'dry',
    'Snow': 'snowy',
    'Scattered Showers': 'wet',
    'Party Cloudy': 'damp',
    'Clear skies': 'dry',
    'Rain likely, temps in low 40s.': 'wet',
    'Hazy': 'damp',
    'Partly Clouidy': 'damp',
    'Sunny Skies': 'dry',
    'Overcast': 'damp',
    'Cloudy, 50% change of rain': 'damp',
    'Fair': 'dry',
    'Light Rain': 'wet',
    'Partly clear': 'dry',
    'Mostly Coudy': 'damp',
    '10% Chance of Rain': 'wet',
    'Cloudy, chance of rain': 'damp',
    'Heat Index 95': 'dry',
    'Sunny, highs to upper 80s': 'dry',
    'Sun & clouds': 'dry',
    'Heavy lake effect snow': 'snowy',
    'Mostly sunny': 'dry',
    'Cloudy, Rain': 'wet',
    'Sunny, Windy': 'dry',
    'Mostly Sunny Skies': 'dry',
    'Rainy': 'wet',
    '30% Chance of Rain': 'wet',
    'Cloudy, light snow accumulating 1-3"': 'snowy',
    'cloudy': 'damp',
    'Clear and Sunny': 'dry',
    'Coudy': 'damp',
    'Clear and sunny': 'dry',
    'Clear to Partly Cloudy': 'dry',
    'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.': 'wet',
    'Rain shower': 'wet',
    'Cold': 'damp'
}

# Step 4: Apply the mapping to create a new categorical variable
df['Weather_Category'] = df[weather_column].map(weather_mapping)

position_mapping = {
    'Quarterback': 'Offence',
    'Wide Receiver': 'Offence',
    'Linebacker': 'Defence',
    'Running Back': 'Offence',
    'Defensive Lineman': 'Defence',
    'Tight End': 'Offence',
    'Safety': 'Defence',
    'Cornerback': 'Defence',
    'Offensive Lineman': 'Offence',
    'Kicker': 'Special'  # Assuming the kicker is mapped to offence as they primarily contribute to scoring
}

# Map positions to Offence or Defence and create a new column 'Off_Def'
df['Off_Def'] = df['RosterPosition'].map(position_mapping)

    
# Feature engineering: Extract game number from GameID
df['GameID'] = df['GameID'].apply(lambda x: int(x.split('-')[1])) 

# Apply the categorization function
df['StadiumCategory'] = df['StadiumType'].apply(categorize_stadium)

# Modify the 'Days_Out' column directly
df['Days_Out'] = df.groupby(['PlayerKey', 'GameID'])['Days_Out'].transform(lambda x: x.max() if x.sum() > 0 else 0)

# Group by and aggregate the rest of the data
aggregated_data = df.groupby(['PlayerKey', 'GameID']).agg({
    'PlayKey': 'count',
}).reset_index()

# Merge the aggregated data back to the original DataFrame
df = pd.merge(df, aggregated_data, on=['PlayerKey', 'GameID'], how='left')

# Fill missing values in 'Days_Out' column with 0 (for non-injured players)
df['Days_Out'].fillna(0, inplace=True)

# Drop duplicate rows based on 'PlayerKey' and 'GameID' columns
df.drop_duplicates(['PlayerKey', 'GameID'], inplace=True)

# Rename the aggregated columns
df.rename(columns={'PlayKey': 'NumPlays'}, inplace=True)

# Define the desired column order
desired_columns = ["PlayerKey", "GameID", "PlayKey_y", "FieldType", "PlayType", 
                   "PlayerDay", "PlayerGame", "Position", "PositionGroup", "RosterPosition", "Off_Def", 
                   "StadiumType", "StadiumCategory", "Temperature", "Weather", "Weather_Category", "BodyPart", "Days_Out"]

# Reorder the columns
df = df[desired_columns]

# Replace temperature values with equal to -999 with 70
df['Temperature'] = df['Temperature'].replace(-999, 70)

def create_comprehensive_interactions(df):
    # Weather interactions (same as before)
    for weather_type in ['wet', 'damp', 'dry', 'indoor']:
        for field_type in ['Natural', 'Synthetic']:
            feature_name = f"{field_type}_{weather_type}"
            df[feature_name] = (df['FieldType'] == field_type) & (df['Weather_Category'] == weather_type)

    # Temperature bin creation
    df['Temp_VeryCold'] = (df['Temperature'] < 30)
    df['Temp_Cold'] = (df['Temperature'] >= 30) & (df['Temperature'] < 50)
    df['Temp_Mild'] = (df['Temperature'] >= 50) & (df['Temperature'] < 80)
    df['Temp_Hot'] = (df['Temperature'] >= 80) & (df['Temperature'] < 95)
    df['Temp_VeryHot'] = (df['Temperature'] >= 95)

    # Temperature bin interactions
    for temp_bin in ['Temp_VeryCold', 'Temp_Cold', 'Temp_Mild', 'Temp_Hot', 'Temp_VeryHot']:
        for field_type in ['Natural', 'Synthetic']:
            feature_name = f"{field_type}_{temp_bin}"
            df[feature_name] = (df['FieldType'] == field_type) & df[temp_bin]

    return df

df = create_comprehensive_interactions(df)

# Combine field condition columns
def combine_field_condition(row):
    if row['Natural_wet'] == 1:
        return 'Natural_wet'
    elif row['Synthetic_wet'] == 1:
        return 'Synthetic_wet'
    elif row['Natural_damp'] == 1:
        return 'Natural_damp'
    elif row['Synthetic_damp'] == 1:
        return 'Synthetic_damp'
    elif row['Natural_dry'] == 1:
        return 'Natural_dry'
    elif row['Synthetic_dry'] == 1:
        return 'Synthetic_dry'
    elif row['Natural_indoor'] == 1:
        return 'Natural_dry'
    elif row['Synthetic_indoor'] == 1:
        return 'Synthetic_dry'
    else:
        return 'None'

df['Field_Condition'] = df.apply(combine_field_condition, axis=1)

# Combine temperature bin columns
def combine_temperature_bin(row):
    if row['Temp_VeryCold'] == 1:
        return 'Temp_VeryCold'
    elif row['Temp_Cold'] == 1:
        return 'Temp_Cold'
    elif row['Temp_Mild'] == 1:
        return 'Temp_Mild'
    elif row['Temp_Hot'] == 1:
        return 'Temp_Hot'
    elif row['Temp_VeryHot'] == 1:
        return 'Temp_VeryHot'
    else:
        return 'None'

df['Temperature_Bin'] = df.apply(combine_temperature_bin, axis=1)

# Combine field temperature bin columns
def combine_field_temperature_bin(row):
    if row['Natural_Temp_VeryCold'] == 1:
        return 'Natural_Temp_VeryCold'
    elif row['Synthetic_Temp_VeryCold'] == 1:
        return 'Synthetic_Temp_VeryCold'
    elif row['Natural_Temp_Cold'] == 1:
        return 'Natural_Temp_Cold'
    elif row['Synthetic_Temp_Cold'] == 1:
        return 'Synthetic_Temp_Cold'
    elif row['Natural_Temp_Mild'] == 1:
        return 'Natural_Temp_Mild'
    elif row['Synthetic_Temp_Mild'] == 1:
        return 'Synthetic_Temp_Mild'
    elif row['Natural_Temp_Hot'] == 1:
        return 'Natural_Temp_Hot'
    elif row['Synthetic_Temp_Hot'] == 1:
        return 'Synthetic_Temp_Hot'
    elif row['Natural_Temp_VeryHot'] == 1:
        return 'Natural_Temp_VeryHot'
    elif row['Synthetic_Temp_VeryHot'] == 1:
        return 'Synthetic_Temp_VeryHot'
    else:
        return 'None'

df['Field_Temperature_Bin'] = df.apply(combine_field_temperature_bin, axis=1)

# Drop the original columns
df.drop(['Natural_wet', 'Synthetic_wet', 'Natural_damp', 'Synthetic_damp', 'Natural_dry', 'Synthetic_dry', 'Natural_indoor', 'Synthetic_indoor',
         'Temp_VeryCold', 'Temp_Cold', 'Temp_Mild', 'Temp_Hot', 'Temp_VeryHot',
         'Natural_Temp_VeryCold', 'Synthetic_Temp_VeryCold', 'Natural_Temp_Cold', 
         'Synthetic_Temp_Cold', 'Natural_Temp_Mild', 'Synthetic_Temp_Mild', 
         'Natural_Temp_Hot', 'Synthetic_Temp_Hot', 'Natural_Temp_VeryHot', 
         'Synthetic_Temp_VeryHot'], axis=1, inplace=True)

df['PlayType_FieldCondition'] = df['PlayType'].astype(str) + "_" + df['Field_Condition'].astype(str)

# Calculate the number of days between games for each player
df['DaysRest'] = df.groupby('PlayerKey')['PlayerDay'].diff().fillna(0)

# Calculate the change in workload between consecutive games
df['WorkloadIncrease'] = df.groupby('PlayerKey')['PlayKey_y'].diff().fillna(0)

# Calculate the workload stress as the product of workload increase and inverse of days rest
df['WorkloadStress'] = df['WorkloadIncrease'] * (1 / df['DaysRest']) 

# Function to calculate pre-injury workload for each player
def calculate_pre_injury_workload(df, num_previous_games=3):
    pre_injury_workloads = []
    
    # Group the DataFrame by PlayerKey
    grouped = df.groupby('PlayerKey')
    
    for player_key, group in grouped:
        # Iterate over each row in the group (player's games)
        for idx, row in group.iterrows():
            # Filter the group to include only the player's previous games
            previous_games = group[(group['GameID'] < row['GameID'])]
            
            # Sort the previous games by GameID in descending order to get the most recent games first
            previous_games = previous_games.sort_values(by='GameID', ascending=False)
            
            # Take the most recent 'num_previous_games' games
            recent_games = previous_games.head(num_previous_games)
            
            # Calculate the player's workload for these games (e.g., sum of plays)
            pre_injury_workload = recent_games['PlayKey_y'].sum()
            
            # Append the calculated pre-injury workload to the list
            pre_injury_workloads.append(pre_injury_workload)
    
    return pre_injury_workloads

# Calculate pre-injury workload for every player
pre_injury_workloads = calculate_pre_injury_workload(df)

# Add pre-injury workload values to the DataFrame
df['PreInjuryWorkload'] = pre_injury_workloads

def calculate_workload_density(df, num_previous_games=3):
    workload_densities = []

    grouped = df.groupby('PlayerKey')

    for player_key, group in grouped:
        for idx, row in group.iterrows():
            previous_games = group[(group['GameID'] < row['GameID']) & (group['PlayerDay'] < row['PlayerDay'])].sort_values(by='GameID', ascending=False).head(num_previous_games)

            pre_injury_workload = previous_games['PlayKey_y'].sum()
            num_days_span = previous_games['PlayerDay'].max() - previous_games['PlayerDay'].min() + 1  # +1 to include both end-dates

            workload_density = pre_injury_workload / num_days_span
            workload_densities.append(workload_density)

    return workload_densities

# Calculate workload density
workload_densities = calculate_workload_density(df)

# Add it as a new column
df['WorkloadDensity'] = workload_densities 

# Create a new column 'Season' to identify the start of a new season for each player
df['Season'] = (df.groupby('PlayerKey')['PlayerDay'].diff() > 100) | (df['PlayerKey'] != df['PlayerKey'].shift())
df['Season'] = (df.groupby('PlayerKey')['Season'].cumsum()).astype(int)  # Reset season count for each new player

def add_game_number_in_season(df):
    df['GameNumberInSeason'] = df.groupby(['PlayerKey', 'Season'])['GameID'].transform(lambda x: x.rank(method='dense').astype(int))
    return df 

df = add_game_number_in_season(df) # Add the new column 'GameNumberInSeason'

df['WorkloadDensity_SeasonAdjusted'] = df['WorkloadDensity'] * (df['GameNumberInSeason'] / df['Season'].max())

def fill_nulls_with_zero(df, columns_for_zero_fill):
    """Fills null values with zero in specified columns.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns_for_zero_fill (list): List of column names where null values should be replaced with zero.

    Returns:
        pandas.DataFrame: The DataFrame with null values replaced by zero.
    """

    # Replace null values with 0 in the specified columns
    df[columns_for_zero_fill] = df[columns_for_zero_fill].fillna(0)

    return df

# Columns where we want to replace nulls with zero
columns_for_zero_fill = ['WorkloadDensity', 'WorkloadStress']

# Replace null values with zero in the specified columns
df = fill_nulls_with_zero(df, columns_for_zero_fill)



# Save the DataFrame to a new CSV file
df.to_csv('Aggregated_Data2.csv', index=False)