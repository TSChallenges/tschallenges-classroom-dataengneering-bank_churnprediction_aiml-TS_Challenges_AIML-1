# tests/test_model_training.py

import pytest
import pandas as pd
from src.model_training import load_processed_data, define_features_target, train_model, evaluate_model
from sklearn.metrics import accuracy_score, f1_score

def test_load_processed_data():
    df = load_processed_data('data/processed_bank_churn.csv')
    assert isinstance(df, pd.DataFrame), "Processed data should be loaded into a pandas DataFrame"
    assert not df.empty, "Processed DataFrame should not be empty"

def test_define_features_target():
    df = load_processed_data('data/processed_bank_churn.csv')
    X, y = define_features_target(df)
    assert isinstance(X, pd.DataFrame), "Features should be a pandas DataFrame"
    assert isinstance(y, pd.Series), "Target should be a pandas Series"
    assert 'Exited' in y.name, "Target column should be 'Exited'"

def test_train_model():
    df = load_processed_data('data/processed_bank_churn.csv')
    X, y = define_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert hasattr(model, 'predict'), "Model should have a predict method"

def test_evaluate_model():
    df = load_processed_data('data/processed_bank_churn.csv')
    X, y = define_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy, f1 = evaluate_model(model, X_test, y_test)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert 0 <= f1 <= 1, "F1 Score should be between 0 and 1"
