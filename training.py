import pandas as pd
from sklearn.model_selection import train_test_split

# Read the merged dataset into a DataFrame
merged_df = pd.read_csv('Aggregated_Data2.csv')

# Create a binary target variable 'Injured' based on the 'Days_Out' column
merged_df['Injured'] = merged_df['Days_Out'].apply(lambda x: 1 if x > 0 else 0)

# Features (input variables)
X = merged_df.drop(['BodyPart', 'Days_Out', 'Injured'], axis=1)

# Target variable
y = merged_df['Injured']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the training and testing sets to CSV files
X_train.to_csv('X_train_classification.csv', index=False)
X_test.to_csv('X_test_classification.csv', index=False)
y_train.to_csv('y_train_classification.csv', index=False, header=True)
y_test.to_csv('y_test_classification.csv', index=False, header=True)