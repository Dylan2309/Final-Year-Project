# Highest Recall and Accuracy for an XGB model
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

x_filepath = 'X_train_classification.csv'
y_filepath = 'y_train_classification.csv'

# Function to read data
def read_data(x_filepath, y_filepath):
    X = pd.read_csv(x_filepath)[['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest']]
    y = pd.read_csv(y_filepath)['Injured']
    return X, y


def preprocess_data(X, y):
    print("Unique values in FieldType column:", X['FieldType'].unique())
    X = pd.get_dummies(X, columns=['PlayType', 'FieldType'], drop_first=False)
    print("Columns after preprocessing:", X.columns.tolist())
    return X, y

# Preprocess the data
X, y = read_data(x_filepath, y_filepath)
X, y = preprocess_data(X, y)

# Print the final order of features before training
print("Final feature order before training:", X.columns.tolist())

# Undersample the data
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Define the model
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Define the parameter grid for RandomizedSearch
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [150, 200, 250, 400],
    'subsample': [0.65, 0.7, 0.85, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 5, 10]
}

# Before fitting the model, especially if using a wrapped model like RandomizedSearchCV
print("Feature names used in training:", X_resampled.columns.tolist())

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=xgb_classifier, param_distributions=param_grid, n_iter=100, scoring='roc_auc', cv=3, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_resampled, y_resampled)

# Best estimator found by RandomizedSearch
xgb_classifier = random_search.best_estimator_

# Output best parameters
print("Best parameters found: ", random_search.best_params_)
joblib.dump(random_search, 'xgboost_search.joblib')  # Save the complete object

# Training the classifier
xgb_classifier.fit(X_resampled, y_resampled)
print("Feature names after fitting the model:", list(X_resampled.columns))
joblib.dump(xgb_classifier, 'xgboost_model.joblib')  # Save the model

# Feature importances
feature_importances = xgb_classifier.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_resampled.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)

# Evaluate the classifier using k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Load the RandomizedSearchCV object and retrieve the best estimator
loaded_search = joblib.load('xgboost_search.joblib')
best_estimator = loaded_search.best_estimator_


# Load the test data (assuming the same preprocessing as the training data)
X_test = pd.read_csv('X_test_classification.csv')[['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest']]
y_test = pd.read_csv('y_test_classification.csv')['Injured']

# Preprocess the test data
X_test = pd.get_dummies(X_test, columns=['PlayType', 'FieldType'], drop_first=True)

# Align test data columns with training data
missing_cols = set(X_resampled.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0
X_test = X_test[X_resampled.columns]

# Load the trained model
best_estimator = joblib.load('xgboost_search.joblib')

# Predict the responses for the test data
y_pred = xgb_classifier.predict(X_test)
y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

# Calculate the performance on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)
print("Classification Report on test set:")
print(classification_report(y_test, y_pred))

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print("AUC on test set:", roc_auc)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix on test set:")
print(conf_matrix)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
