# tests/test_eda.py

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from src.eda import load_data, check_missing_values, generate_summary_statistics, visualize_distributions

def test_load_data():
    df = load_data('data/bank_churn.csv')
    assert isinstance(df, pd.DataFrame), "Data should be loaded into a pandas DataFrame"
    assert not df.empty, "DataFrame should not be empty"

def test_missing_values():
    df = load_data('data/bank_churn.csv')
    missing = df.isnull().sum()
    assert missing.sum() == 0, "There should be no missing values in the dataset"

def test_summary_statistics():
    df = load_data('data/bank_churn.csv')
    summary = df.describe()
    assert 'Age' in summary.columns, "Summary statistics should include 'Age'"
    assert 'Balance' in summary.columns, "Summary statistics should include 'Balance'"

def test_visualizations():
    df = load_data('data/bank_churn.csv')
    # Since visualize_distributions() shows plots, we can check if it runs without errors
    try:
        visualize_distributions(df)
    except Exception as e:
        pytest.fail(f"Visualization failed with exception: {e}")
