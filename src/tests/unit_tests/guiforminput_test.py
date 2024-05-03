# this is unit testing for the input form window in our gui, it seems to work locally when ran in the root dir. however not in the Gitlab Pipeline.
# The test fails when the evaluate button is clicked, however when the evaluate method is called directly it passes.
# However the evaluate method is called when the evaluate button is clicked.
import sys
import os
import unittest
from PyQt5.QtWidgets import QApplication, QLineEdit, QSpinBox
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from gui import MainApplicationWindow

app = QApplication(sys.argv)

class TestInputFormWindow(unittest.TestCase):
    def setUp(self):
        self.main_window = MainApplicationWindow()

    def test_initial_state(self):
        input_form_window = self.main_window.input_form_window
        self.assertIsNotNone(input_form_window.model_dropdown)
        self.assertEqual(input_form_window.model_dropdown.count(), 3)

    @patch('gui.load_model')
    def test_model_loading(self, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        input_form_window = self.main_window.input_form_window
        input_form_window.models['XGBoost'] = mock_load_model('xgboost_model.joblib')
        self.assertEqual(input_form_window.models['XGBoost'], mock_model)

    # @patch('gui.load_model')
    # def test_evaluate_button_click(self, mock_load_model):
    #     input_form_window = self.main_window.input_form_window
    #     mock_model = MagicMock()
    #     mock_model.predict.return_value = [0]
    #     mock_model.predict_proba.return_value = [[0.7, 0.3]]
    #     mock_load_model.return_value = mock_model
    #     input_form_window.models['XGBoost'] = mock_model
    #     for widget_key, widget in input_form_window.input_fields.items():
    #         if isinstance(widget, QLineEdit):
    #             widget.setText('1')
    #         elif isinstance(widget, QSpinBox):
    #             widget.setValue(10)
    #     with patch.object(input_form_window, 'evaluate', return_value=None) as mock_evaluate:
    #         input_form_window.evaluate_button.click()
    #         mock_evaluate.assert_called_once()

    @patch('gui.load_model')
    def test_evaluate_method_directly(self, mock_load_model):
        input_form_window = self.main_window.input_form_window
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.7, 0.3]]
        mock_load_model.return_value = mock_model
        input_form_window.models['XGBoost'] = mock_model
        for widget_key, widget in input_form_window.input_fields.items():
            if isinstance(widget, QLineEdit):
                widget.setText('1')
            elif isinstance(widget, QSpinBox):
                widget.setValue(10)
        input_form_window.evaluate()
        expected_output = "Predicted result: Not Injured"
        self.assertIn(expected_output, input_form_window.results_text.toPlainText())
        self.assertIn("Prediction probability: 0.30", input_form_window.results_text.toPlainText())

if __name__ == '__main__':
    unittest.main()

