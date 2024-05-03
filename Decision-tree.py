import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler

def load_data():
    X = pd.read_csv('X_train_classification.csv')[['PlayKey_y','FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest']]
    y = pd.read_csv('y_train_classification.csv')['Injured']
    return X, y

def preprocess_data(X):
    X = pd.get_dummies(X, columns=['PlayType', 'FieldType'], drop_first=False)
    return X

# Read and preprocess the data
X, y = load_data()
X = preprocess_data(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply random undersampling to balance the classes
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Define the parameter grid
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 4, 20],
    'min_samples_leaf': [3, 4, 8],
    'criterion': ['gini', 'entropy']
}

# Create the grid search object
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1_macro',
    verbose=1
)

# Fit the grid search to the data
grid_search.fit(X_train_resampled, y_train_resampled)

# Save the complete GridSearchCV object
joblib.dump(grid_search, 'decision_tree_search.joblib')

# Load the GridSearchCV object
loaded_search = joblib.load('decision_tree_search.joblib')

# Get the best parameters and create a classifier with them
best_clf = DecisionTreeClassifier(**loaded_search.best_params_, random_state=42, class_weight='balanced')
best_clf.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_prob = best_clf.predict_proba(X_test)[:, 1]

# Adjust the threshold
new_threshold = 0.55
y_pred_adjusted = (y_prob > new_threshold).astype(int)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Evaluate the model with the adjusted threshold
f1_adjusted = f1_score(y_test, y_pred_adjusted, average='macro')
precision_adjusted = precision_score(y_test, y_pred_adjusted, average='macro')
recall_adjusted = recall_score(y_test, y_pred_adjusted, average='macro')
conf_matrix_adjusted = confusion_matrix(y_test, y_pred_adjusted)
class_report_adjusted = classification_report(y_test, y_pred_adjusted)

# Print the evaluation metrics with the adjusted threshold
print("Best Parameters:", loaded_search.best_params_)
print("Best Score:", loaded_search.best_score_)
print(f"F1-score (Adjusted Threshold, macro-average): {f1_adjusted:.4f}")
print(f"Precision (Adjusted Threshold, macro-average): {precision_adjusted:.4f}")
print(f"Recall (Adjusted Threshold, macro-average): {recall_adjusted:.4f}")
print("Confusion Matrix (Adjusted Threshold):")
print(conf_matrix_adjusted)
print("Classification Report (Adjusted Threshold):")
print(class_report_adjusted)

# Plot feature importances
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
