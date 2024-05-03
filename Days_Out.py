import pandas as pd

# Read the CSV file into a DataFrame
injury_df = pd.read_csv('InjuryRecord.csv')

# Function to calculate Days_Out
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
injury_df['Days_Out'] = injury_df.apply(calculate_days_out, axis=1)

# Drop the individual 'DM' columns
injury_df = injury_df.drop(['DM_M1', 'DM_M7', 'DM_M28', 'DM_M42'], axis=1)

# Save the modified DataFrame to a new CSV file
injury_df.to_csv('InjuryRecordWithDaysOut.csv', index=False)
