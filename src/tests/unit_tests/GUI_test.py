from sre_constants import ANY
import sys
import pandas as pd
import numpy as np
import joblib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QComboBox, QLabel, QTextEdit, QFormLayout, QSpinBox, QLineEdit, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QHeaderView
from PyQt5.QtGui import QIntValidator
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor

import pytest
from gui import MainApplicationWindow, ModelEvaluationWindow, FilterSearchWindow, InputFormWindow
from unittest.mock import patch, MagicMock
from gui import load_data, load_model, preprocess_data

def test_load_data():
    # Create a sample DataFrame for X
    sample_X = pd.DataFrame({
        'PlayKey_y': [1, 2],
        'FieldType': ['Synthetic', 'Natural'],
        'PlayType': ['Pass', 'Rush'],
        'WorkloadDensity': [0.5, 0.7],
        'WorkloadStress': [10, 15],
        'Temperature': [98, 67],
        'WorkloadIncrease': [5, 3],
        'DaysRest': [2, 4]
    })
    
    # Create a sample DataFrame for y
    sample_y = pd.DataFrame({'Injured': [0, 1]})

    # Mock the read_csv function to return the sample data
    with patch('pandas.read_csv', side_effect=lambda file_name: sample_X if 'X_train' in file_name else sample_y):
        X, y = load_data()  # Call the function to be tested
    
        # Check if the returned data matches the sample data
        pd.testing.assert_frame_equal(X, sample_X)
        pd.testing.assert_series_equal(y, sample_y['Injured'])

def test_load_model_success():
    # Create a MagicMock to simulate a loaded model object
    mock_model = MagicMock()

    # Use patch to simulate successful model loading
    with patch('joblib.load', return_value=mock_model):
        model = load_model('fake_model_path.joblib')
        assert model == mock_model, "Model should match the mock model object when loaded successfully"

def test_load_model_failure():
    # exception is raised when loading the model
    with patch('joblib.load', side_effect=Exception("Failed to load model")):
        model = load_model('non_existent_model_path.joblib')
        assert model is None, "Model should be None when loading fails"

@pytest.fixture
def app(qtbot):
    """Fixture to create an application for the tests."""
    return QApplication([])

def test_main_application_window_initialization(app, qtbot):
    window = MainApplicationWindow()
    qtbot.addWidget(window)  # addinf the window to the qtbot

    # Check the window title and geometry
    assert window.windowTitle() == 'Model Evaluation and Data Input Platform'
    assert window.geometry() == QRect(100, 100, 1000, 1500)  # Check if the window size and position are correct

    # Verify the correct number of tabs
    assert window.tab_widget.count() == 3

    # Check the labels and content of the tabs
    assert window.tab_widget.tabText(0) == 'Model Evaluation'
    assert window.tab_widget.tabText(1) == 'Filter/Search Data'
    assert window.tab_widget.tabText(2) == 'Input and Evaluate'
    assert isinstance(window.tab_widget.widget(0), ModelEvaluationWindow)
    assert isinstance(window.tab_widget.widget(1), FilterSearchWindow)
    assert isinstance(window.tab_widget.widget(2), InputFormWindow)

@pytest.fixture
def model_evaluation_window(qtbot):
    from gui import ModelEvaluationWindow
    window = ModelEvaluationWindow(model_paths={})
    qtbot.addWidget(window)
    return window

def test_initialization(model_evaluation_window):
    assert model_evaluation_window.model_dropdown.count() == 3  # Expecting three items
    assert model_evaluation_window.model_dropdown.itemText(0) == "XGBoost"
    assert model_evaluation_window.model_dropdown.itemText(1) == "Decision Tree"
    assert model_evaluation_window.model_dropdown.itemText(2) == "Random Forest"
    assert model_evaluation_window.message_area.isReadOnly()  # Should be True, corrected assertion


def test_connections(model_evaluation_window, qtbot):
    with qtbot.waitSignal(model_evaluation_window.load_button.clicked, timeout=1000) as blocker:
        qtbot.mouseClick(model_evaluation_window.load_button, Qt.LeftButton)
    assert blocker.signal_triggered  # Check if the signal was emitted


def test_layouts(model_evaluation_window):
    # Verifies that the main layout is a QVBoxLayout
    assert isinstance(model_evaluation_window.layout(), QVBoxLayout)
    # Check if the summary tab has the correct layout set
    assert isinstance(model_evaluation_window.summary_tab.layout(), QVBoxLayout)

def setupFeatureImportancesTab(self):
    self.feature_importances_tab.setLayout(QVBoxLayout())
    self.feature_importances_figure = Figure()
    self.feature_importances_canvas = FigureCanvas(self.feature_importances_figure)
    self.feature_importances_tab.layout().addWidget(self.feature_importances_canvas)


def setupConfMatrixTab(self):
    self.conf_matrix_tab.setLayout(QVBoxLayout())
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.conf_matrix_tab.layout().addWidget(self.canvas)

def setupROCCurveTab(self):
    self.roc_curve_tab.setLayout(QVBoxLayout())
    self.roc_figure = Figure()
    self.roc_canvas = FigureCanvas(self.roc_figure)
    self.roc_curve_tab.layout().addWidget(self.roc_canvas)

from unittest.mock import patch, MagicMock

def test_load_and_evaluate_data_success(model_evaluation_window, qtbot):
    with patch('gui.load_data') as mock_load_data, \
         patch('gui.preprocess_data') as mock_preprocess, \
         patch('gui.ModelEvaluationWindow.evaluate_model') as mock_evaluate_model:
        
        mock_load_data.return_value = (MagicMock(), MagicMock())  # Assume this is the data
        mock_preprocess.return_value = MagicMock()  # Assume this is the preprocessed data
        mock_evaluate_model.return_value = None  # Depends on what evaluate_model does

        model_evaluation_window.load_and_evaluate_data()

        # Check if the functions were called
        mock_load_data.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_evaluate_model.assert_called_once()

def test_load_and_evaluate_data_failure(model_evaluation_window, qtbot):
    with patch('gui.load_data', side_effect=Exception("Test error")), \
         patch.object(model_evaluation_window, 'summary_text', new_callable=MagicMock()) as mock_summary_text:
        
        model_evaluation_window.load_and_evaluate_data()

        mock_summary_text.setText.assert_called_with("Failed to preprocess or evaluate data: Test error")

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })

def test_evaluate_model_success(model_evaluation_window, sample_dataframe):
    with patch('gui.load_model') as mock_load_model, \
         patch('gui.ModelEvaluationWindow.display_results') as mock_display_results:
        mock_search = MagicMock()
        mock_search.best_estimator_ = MagicMock()
        mock_search.best_estimator_.predict = MagicMock(return_value=np.array([1]))
        mock_search.best_estimator_.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9]]))
        mock_search.best_params_ = {'param': 'value'}
        mock_search.best_score_ = 0.9
        mock_load_model.return_value = mock_search

        model_evaluation_window.evaluate_model(sample_dataframe, None)  # Call the method under test

        mock_load_model.assert_called_once()
        mock_display_results.assert_called_once()


def test_evaluate_model_failure(model_evaluation_window):
    with patch('gui.load_model', return_value=None), \
         patch.object(model_evaluation_window, 'summary_text') as mock_summary_text:
        model_evaluation_window.evaluate_model(None, None)
        mock_summary_text.setText.assert_called_once_with("Failed to load the model.")

def test_display_results_basic(model_evaluation_window):
    # Create some sample data
    X = np.random.rand(10, 5)  # Random features
    y = np.random.randint(0, 2, size=10)  # Random labels
    y_pred = np.random.randint(0, 2, size=10)  # Random predictions

    # Create a mock object to simulate the search object
    search_mock = MagicMock()
    search_mock.best_estimator_.predict_proba = MagicMock(return_value=np.random.rand(10, 2))

    # Call the method under test
    with patch('matplotlib.axes.Axes.plot') as mock_plot:
        model_evaluation_window.display_results(X, y, y_pred, {}, 0.85, {}, search_mock)

        # Check that plot was called
        mock_plot.assert_called()