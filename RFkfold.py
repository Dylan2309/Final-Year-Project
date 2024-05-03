import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import randint

x_filepath = 'X_train_classification.csv'
y_filepath = 'y_train_classification.csv'

#Function to read data
def read_data(x_filepath, y_filepath):
    X = pd.read_csv(x_filepath)[['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest']]
    y = pd.read_csv(y_filepath)['Injured']
    return X, y


def preprocess_data(X, y):
    print("Unique values in FieldType column:", X['FieldType'].unique())
    X = pd.get_dummies(X, columns=['PlayType', 'FieldType'], drop_first=False)
    print("Columns after preprocessing:", X.columns.tolist())
    return X, y


#Preprocess the data
X, y = read_data(x_filepath, y_filepath)
X, y = preprocess_data(X, y)

# Print the final order of features before training
print("Final feature order before training:", X.columns.tolist())


# Undersample the data
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Define hyperparameter distributions for RandomizedSearchCV
param_distributions = {
    'max_depth': randint(8, 10),
    'n_estimators': randint(100, 750),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}

# Before fitting the model, especially if using a wrapped model like RandomizedSearchCV
print("Feature names used in training:", X_resampled.columns.tolist())

# Define RandomForest classifier with balanced class weights
rf_classifier = RandomForestClassifier(class_weight={0: 1, 1: 1.141}, random_state=42)

# Use RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_distributions,
                                   n_iter=100, scoring='accuracy',
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_resampled, y_resampled)

# Save the complete RandomizedSearchCV object
joblib.dump(random_search, 'random_forest_search.joblib')

# Load the RandomizedSearchCV object
loaded_search = joblib.load('random_forest_search.joblib')

# Print best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Retrieve the best estimator and its metrics
best_estimator = loaded_search.best_estimator_
best_params = loaded_search.best_params_
best_score = loaded_search.best_score_

# Feature importances
feature_importances = best_estimator.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)

# Evaluate using the best estimator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    best_estimator.fit(X_train, y_train)
    y_pred = best_estimator.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Saving the trained model
joblib.dump(best_estimator, 'random_forest_model.joblib')

# Load the model for testing
best_estimator = joblib.load('random_forest_model.joblib')

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
best_estimator = joblib.load('random_forest_model.joblib')

# Predict the responses for the test data
y_pred = best_estimator.predict(X_test)
y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]

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
from sklearn.metrics import confusion_matrix
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