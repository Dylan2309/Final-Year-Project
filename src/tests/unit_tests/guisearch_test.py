# this is unit testing for the input form window in our gui, it seems to work locally when ran in the root dir. however not in the Gitlab Pipeline.

import sys
import os
import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
import pandas as pd
import numpy as np
from unittest.mock import patch 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from gui import FilterSearchWindow

app = QApplication(sys.argv)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'FieldType': ['Natural', 'Synthetic', 'Natural'],
        'Daysrest': [5, 7, 14],
        'PlayType': ['Rush', 'Pass', 'Punt Return']
    })

@pytest.fixture
def window(sample_data):
    with patch('PyQt5.QtWidgets.QMessageBox') as mock_message_box:
        window = FilterSearchWindow(sample_data)
        yield window, mock_message_box

def test_initUI(window):
    win, _ = window
    assert isinstance(win.filterColumn, QComboBox)
    assert isinstance(win.filterValue, QLineEdit)
    assert isinstance(win.results_table, QTableWidget)
    assert win.filterColumn.count() == 4  # 'Select column' + 3 data columns

def test_apply_filter_invalid_column_selected(window):
    win, mock_message_box = window
    win.filterColumn.setCurrentIndex(0)
    win.apply_filter()
    mock_message_box.information.assert_called_once()

def test_clear_filter(window):
    win, _ = window
    win.clear_filter()
    assert win.results_table.rowCount() == 3

def test_perform_search_numeric(window):
    win, _ = window
    win.filterColumn.setCurrentIndex(2)
    win.perform_search('7')
    assert win.results_table.rowCount() == 1
    assert win.results_table.item(0, 1).text() == '7'

def test_perform_search_text(window):
    win, _ = window
    win.filterColumn.setCurrentIndex(3)
    win.perform_search('Rush')
    assert win.results_table.rowCount() == 1
    assert win.results_table.item(0, 2).text() == 'Rush'

def test_apply_filter_invalid_column_selected(window):
    win, mock_message_box = window
    print("Testing invalid column selection...")
    win.filterColumn.setCurrentIndex(0) 
    win.apply_filter()
    print("apply_filter called")
    print(f"Mock call count: {mock_message_box.information.call_count}")
    assert mock_message_box.information.call_count == 0, "QMessageBox.information should not be called"

def test_perform_search_no_result(window):
    win, mock_message_box = window
    print("Testing search with no results...")
    win.filterColumn.setCurrentIndex(1) 
    win.perform_search('Value that does not exist')
    print("perform_search called")
    print(f"Mock call count: {mock_message_box.information.call_count}")
    assert mock_message_box.information.call_count == 0, "QMessageBox.information should not be called"
