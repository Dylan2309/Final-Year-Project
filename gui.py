import sys
import pandas as pd
import numpy as np
import re
import joblib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QComboBox, QLabel, QTextEdit, QFormLayout, QSpinBox, QLineEdit, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QHeaderView, QGroupBox
from PyQt5.QtGui import QIntValidator
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


def load_data():
    X = pd.read_csv('X_train_classification.csv')[['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest']]
    y = pd.read_csv('y_train_classification.csv')['Injured']
    return X, y

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return None

# Function to preprocess data
def preprocess_data(X):
    X = pd.get_dummies(X, columns=['PlayType', 'FieldType'], drop_first=False)
    return X

class MainApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Evaluation and Data Input Platform')
        self.setGeometry(100, 100, 1000, 1500)
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        self.data, _ = load_data()
        
        # Define model paths and features
        model_paths = {
            'XGBoost': 'xgboost_model.joblib',
            'Decision Tree': 'decision_tree_search.joblib',
            'Random Forest': 'random_forest_model.joblib'
        }
        # Define model features
        model_features = {
            'XGBoost': ['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest'],
            'Decision Tree': ['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest'],
            'Random Forest': ['PlayKey_y', 'FieldType', 'PlayType', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 'DaysRest']
        }
        
        self.model_evaluation_window = ModelEvaluationWindow(model_paths) # Pass the model paths
        self.filter_search_window = FilterSearchWindow(self.data)
        self.input_form_window = InputFormWindow(model_paths, model_features)
        
        self.tab_widget.addTab(self.model_evaluation_window, "Model Evaluation")
        self.tab_widget.addTab(self.filter_search_window, "Filter/Search Data")
        self.tab_widget.addTab(self.input_form_window, "Input and Evaluate")

    def onTabChange(self, index):
        print("Tab changed to", index)

class ModelEvaluationWindow(QWidget): # This class is used to evaluate the model
    def __init__(self, model_paths):
        super().__init__()
        self.initUI()
        self.X, self.y = None, None

    def initUI(self): # This function initializes the UI
        layout = QVBoxLayout(self)
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["XGBoost", "Decision Tree", "Random Forest"]) # Add the models to the dropdown
        layout.addWidget(self.model_dropdown)

        self.load_button = QPushButton('Load and Evaluate Data') # Create a button to load and evaluate the data
        self.load_button.clicked.connect(self.load_and_evaluate_data)
        layout.addWidget(self.load_button)

        # Add a tab widget to display the results
        self.tabs = QTabWidget()
        self.summary_tab = QWidget()
        self.feature_importances_tab = QWidget()
        self.conf_matrix_tab = QWidget()
        self.roc_curve_tab = QWidget()

        # Add the tabs to the tab widget
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.feature_importances_tab, "Feature Importances")
        self.tabs.addTab(self.conf_matrix_tab, "Confusion Matrix")
        self.tabs.addTab(self.roc_curve_tab, "ROC Curve") 


        layout.addWidget(self.tabs)

        # Set up each tab
        self.setupSummaryTab()
        self.setupFeatureImportancesTab()
        self.setupConfMatrixTab()
        self.setupROCCurveTab()

        self.message_area = QTextEdit()
        self.message_area.setReadOnly(True)
        layout.addWidget(self.message_area)
        self.message_area.setMaximumHeight(100)
        
        self.summary_text = QTextEdit(self)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        self.summary_text.setMaximumHeight(100)

    def setupSummaryTab(self): # This function sets up the summary tab
        layout = QVBoxLayout()
        self.summary_table = QTableWidget()
        self.summary_table.setRowCount(7)  # Number of different metrics to display
        self.summary_table.setColumnCount(2)
        self.summary_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.summary_table)
        self.summary_tab.setLayout(layout)


    def setupFeatureImportancesTab(self): # This function sets up the feature importances tab
        layout = QVBoxLayout()
        self.feature_importances_figure = Figure()
        self.feature_importances_canvas = FigureCanvas(self.feature_importances_figure)
        layout.addWidget(self.feature_importances_canvas)
        self.feature_importances_tab.setLayout(layout)


    def setupConfMatrixTab(self): # This function sets up the confusion matrix tab
        layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.conf_matrix_tab.setLayout(layout)

    def setupROCCurveTab(self): # This function sets up the ROC curve tab
        layout = QVBoxLayout()
        self.roc_figure = Figure()
        self.roc_canvas = FigureCanvas(self.roc_figure)
        layout.addWidget(self.roc_canvas)
        self.roc_curve_tab.setLayout(layout)


    def load_and_evaluate_data(self): # This function loads and evaluates the data
        try:
            self.load_data()
            X_preprocessed = preprocess_data(self.X)
            self.evaluate_model(X_preprocessed, self.y)
        except Exception as e:
            self.summary_text.setText(f"Failed to preprocess or evaluate data: {str(e)}")

    def load_data(self): # This function loads the data
        self.X, self.y = load_data()
        print("Data loaded successfully.")

    def evaluate_model(self, X, y): # This function evaluates the model
        model_name = self.model_dropdown.currentText().lower().replace(" ", "_")
        model_path = f"{model_name}_search.joblib"
        search = load_model(model_path)  # This loads the model
        if search:
            model = search.best_estimator_
            y_pred = model.predict(X)
            best_params = getattr(search, 'best_params_', 'Not available')
            best_score = getattr(search, 'best_score_', 'Not available')
            feature_importances = dict(zip(X.columns, getattr(model, 'feature_importances_', [])))
            self.display_results(X, y, y_pred, best_params, best_score, feature_importances, search)
        else:
            self.summary_text.setText("Failed to load the model.") # Display an error message if the model fails to load

    def display_results(self, X, y, y_pred, best_params, best_score, feature_importances, search): # This function displays the results
        f1 = f1_score(y, y_pred, average='macro')
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro')
        recall = precision_score(y, y_pred, average='macro')
        conf_matrix = confusion_matrix(y, y_pred)

        # Calculating ROC Curve and AUC
        y_proba = search.best_estimator_.predict_proba(X)[:, 1]  
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC Curve
        self.roc_figure.clear()
        ax_roc = self.roc_figure.add_subplot(111)
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
        self.roc_canvas.draw()

        # Display the results in the summary tab
        metrics = [
            ("F1 Score", f"{f1:.4f}"),
            ("Accuracy", f"{accuracy:.4f}"),
            ("Precision", f"{precision:.4f}"),
            ("Recall", f"{recall:.4f}"),
            ("Best Score", f"{best_score:.4f}")
        ]

        # Adding best parameters as separate rows
        for key, value in best_params.items():
            metrics.append((f"Param: {key}", str(value)))

        self.summary_table.setRowCount(len(metrics))
        for row, (metric, value) in enumerate(metrics):
            self.summary_table.setItem(row, 0, QTableWidgetItem(metric))
            self.summary_table.setItem(row, 1, QTableWidgetItem(value))

        # Plot the confusion matrix
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sns.heatmap(conf_matrix, ax=ax, annot=True, fmt='d', cmap='coolwarm', square=True)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        self.canvas.draw()

        # Plot the sorted feature importances in the feature importances tab
        self.feature_importances_figure.clear()
        if feature_importances:
            # Sort feature importances by value
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1])
            features, importances = zip(*sorted_features)
            indices = np.arange(len(features))

            ax_fi = self.feature_importances_figure.add_subplot(111)
            ax_fi.barh(indices, importances, align='center', color='skyblue')
            ax_fi.set_yticks(indices)
            ax_fi.set_yticklabels(features)
            ax_fi.set_xlabel("Feature importance")
            ax_fi.set_title("Feature Importances")
            self.feature_importances_canvas.draw()
        else:
            ax_fi = self.feature_importances_figure.add_subplot(111)
            ax_fi.text(0.5, 0.5, 'No feature importances available', horizontalalignment='center', verticalalignment='center', transform=ax_fi.transAxes)
            ax_fi.axis('off')
            self.feature_importances_canvas.draw()

class FilterSearchWindow(QWidget):  
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.filterColumn = QComboBox()
        self.filterColumn.addItems(["Select column"] + list(self.data.columns))
        layout.addWidget(self.filterColumn)
        self.filterValue = QLineEdit()
        layout.addWidget(self.filterValue)

        applyFilterButton = QPushButton('Apply Filter')
        applyFilterButton.clicked.connect(self.apply_filter)
        layout.addWidget(applyFilterButton)

        clearFilterButton = QPushButton('Clear Filter')
        clearFilterButton.clicked.connect(self.clear_filter)
        layout.addWidget(clearFilterButton)

        searchBar = QLineEdit()
        layout.addWidget(searchBar)

        searchButton = QPushButton('Search')
        searchButton.clicked.connect(lambda: self.perform_search(searchBar.text()))
        layout.addWidget(searchButton)

        # Initialize the results table
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.results_table)


    def apply_filter(self):
        column = self.filterColumn.currentText()
        if column == "Select column" or not column:  # Check if a valid column is selected
            QMessageBox.information(self, "Error", "Please select a column to search.")
            return

        value = self.filterValue.text().strip()
        filtered_data = self.data[self.data[column].astype(str).str.contains(value, case=False)]
        self.update_display(filtered_data)

    def clear_filter(self):
        self.update_display(self.data)

    def perform_search(self, search_text):
        selected_column = self.filterColumn.currentText()
        if selected_column == "Select column" or not selected_column:  # Check if a valid column is selected
            QMessageBox.information(self, "Error", "Please select a column to search.")
            return

        try:
            # Try to handle as a number if applicable
            if self.data[selected_column].dtype in [np.int64, np.float64]:
                search_number = float(search_text)  # Use float to handle both integers and decimals
                mask = self.data[selected_column] == search_number
            else:
                raise ValueError("Not a numeric column")
        except (ValueError, TypeError):
             #  Handle as a string if not a number
            mask = self.data[selected_column].astype(str).str.contains(search_text, case=False, na=False)
    
        search_results = self.data[mask]
        self.update_display(search_results)

    def update_display(self, data):
        if data.empty:  # Check if the DataFrame is empty
            QMessageBox.information(self, "Search Result", "Input not found.")
            self.results_table.setRowCount(0)  # Clear previous results if any
            self.results_table.setColumnCount(0)
        else:
            self.results_table.clear()
            self.results_table.setRowCount(len(data))
            self.results_table.setColumnCount(len(data.columns))
            self.results_table.setHorizontalHeaderLabels(data.columns.tolist())

            for row_index, row_data in enumerate(data.itertuples(index=False), start=0):
                for column_index, value in enumerate(row_data):
                    self.results_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))

            self.results_table.resizeColumnsToContents()

class InputFormWindow(QWidget):
    def __init__(self, model_paths, model_features):
        super().__init__()
        self.model_paths = model_paths
        self.model_features = model_features
        self.models = {model: load_model(path) for model, path in model_paths.items()}
        print("Loaded models:", self.models)
        self.model_accuracies = {
            'XGBoost': 69,
            'Decision Tree': 50,
            'Random Forest': 63
        }
        self.setupUI()


    def setupUI(self):
        layout = QVBoxLayout(self)
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(list(self.model_features.keys()))
        self.model_dropdown.currentIndexChanged.connect(self.update_input_fields)
        layout.addWidget(self.model_dropdown)

        # Input fields grouped
        self.input_group_box = QGroupBox("Input Parameters")
        self.form_layout = QFormLayout()
        self.input_fields = {}
        self.update_input_fields()
        self.input_group_box.setLayout(self.form_layout)
        layout.addWidget(self.input_group_box)

        # Evaluate button
        self.evaluate_button = QPushButton('Evaluate')
        self.evaluate_button.clicked.connect(self.evaluate)
        layout.addWidget(self.evaluate_button)

        # Results display
        self.results_group_box = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        self.results_group_box.setLayout(results_layout)
        layout.addWidget(self.results_group_box)
        
        # Set up the figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.results_group_box.setStyleSheet("""
        QGroupBox {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        QTextEdit {
            background-color: #f0f0f0;
            color: #333;
            font-size: 14px;
        }
    """)
        
    def update_graph(self, accuracy):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        labels = 'Accuracy', 'Remaining'
        sizes = [accuracy, 100 - accuracy]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.canvas.draw()


    def update_input_fields(self):
        for widget in self.input_fields.values():
            self.form_layout.removeRow(widget)
        self.input_fields.clear()

        model_key = self.model_dropdown.currentText()
        features = self.model_features[model_key]

        # Move 'WorkloadStress' to the end if it exists in the list
        if 'WorkloadIncrease' in features:
            features.remove('WorkloadIncrease')
            features.append('WorkloadIncrease')

        if 'WorkloadStress' in features:
            features.remove('WorkloadStress')
            features.append('WorkloadStress')
            

        if 'DaysRest' not in features:  # Make sure DaysRest is always there
            features.append('DaysRest')

        for feature in features:
            if feature in ['Temperature', 'DaysRest', 'WorkloadIncrease']:
                input_widget = QSpinBox()
                if feature == 'DaysRest':
                    input_widget.setMinimum(1)  # Set minimum to 1
                    input_widget.setValue(1)     # Set the starting value to 1
                    input_widget.setMaximum(365) # Max days in a year
                elif feature == 'WorkloadIncrease':
                    input_widget.setMinimum(-100)  # Example range that allows negatives
                    input_widget.setMaximum(100)
                input_widget.valueChanged.connect(self.calculate_workload_stress)
            else:
                input_widget = QLineEdit()
                input_widget.setReadOnly(feature == 'WorkloadStress')  # Make WorkloadStress read-only
            self.input_fields[feature] = input_widget
            self.form_layout.addRow(QLabel(feature + ":"), input_widget)

    def calculate_workload_stress(self):
        try:
            workload_increase = self.input_fields['WorkloadIncrease'].value()
            days_rest = self.input_fields['DaysRest'].value()
            workload_stress = workload_increase / days_rest
            self.input_fields['WorkloadStress'].setText(f"{workload_stress:.2f}")
        except KeyError:
            self.results_text.setText("Please ensure all necessary fields are initialized.")


    def local_preprocess(self, input_data): # This function preprocesses the input data
        input_df = pd.DataFrame(input_data)

        print("Initial DataFrame columns:", input_df.columns.tolist()) # Print the initial columns

        required_cols = ['FieldType', 'PlayType', 'Temperature', 'WorkloadIncrease', 'DaysRest', 'WorkloadDensity', 'PlayKey_y'] # Required columns 
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        numeric_features = ['PlayKey_y', 'WorkloadDensity', 'Temperature', 'WorkloadIncrease', 'DaysRest']
        for feature in numeric_features:
            input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0)

        print("Unique FieldType values:", input_df['FieldType'].unique())
        print("Unique PlayType values:", input_df['PlayType'].unique())

        categorical_cols = ['FieldType', 'PlayType']
        input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)

        print("DataFrame columns after adding dummies:", input_df.columns.tolist())

        expected_columns = ['PlayKey_y', 'WorkloadDensity', 'WorkloadStress', 'Temperature', 'WorkloadIncrease', 
                            'DaysRest', 'PlayType_0', 'PlayType_Extra Point', 'PlayType_Field Goal', 'PlayType_Kickoff', 
                            'PlayType_Kickoff Not Returned', 'PlayType_Kickoff Returned', 'PlayType_Pass', 'PlayType_Punt', 
                            'PlayType_Punt Not Returned', 'PlayType_Punt Returned', 'PlayType_Rush', 'FieldType_Natural', 
                            'FieldType_Synthetic']

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        return input_df



    def evaluate(self): # This function evaluates the model
        model_key = self.model_dropdown.currentText()
        model = self.models.get(model_key)

        if not model:
            self.results_text.setText("Selected model is not available. Please check model loading.")
            return

        input_data = {feature: float(widget.text()) if isinstance(widget, QSpinBox) or widget.isReadOnly() else widget.text()
                    for feature, widget in self.input_fields.items() if widget.text()}

        try: # Try to predict the result
            input_df = pd.DataFrame([input_data])
            preprocessed_data = self.local_preprocess(input_df)

            result = model.predict(preprocessed_data)[0]
            prediction_prob = model.predict_proba(preprocessed_data)[0][1] * 100  # as percentage
            model_accuracy = self.model_accuracies.get(model_key, 'Unknown')  # Fetch accuracy dynamically

            result_text = (
                f"Predicted Result: {'Injured' if result == 1 else 'Not Injured'}\n"
                f"Model Used: {model_key}\n"
                f"Model Accuracy: {model_accuracy}%"
            )
            self.results_text.setText(result_text)
            self.update_graph(model_accuracy)

        except Exception as e:
            self.results_text.setText(f"Error in prediction: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainApplicationWindow()
    main_window.show()
    sys.exit(app.exec_())
