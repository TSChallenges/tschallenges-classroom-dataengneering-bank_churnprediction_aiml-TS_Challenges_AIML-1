# tests/test_feature_engineering.py

import pytest
import pandas as pd
from src.feature_engineering import load_data, handle_missing_values, create_age_groups, encode_categorical_features, save_processed_data

def test_handle_missing_values():
    df = load_data('data/bank_churn.csv')
    df = handle_missing_values(df)
    assert not df.isnull().values.any(), "There should be no missing values after handling"

def test_create_age_groups():
    df = load_data('data/bank_churn.csv')
    df = handle_missing_values(df)
    df = create_age_groups(df)
    assert 'Age_Group' in df.columns, "'Age_Group' column should be created"
    assert df['Age_Group'].isnull().sum() == 0, "'Age_Group' should not have missing values"

def test_encode_categorical_features():
    df = load_data('data/bank_churn.csv')
    df = handle_missing_values(df)
    df = create_age_groups(df)
    df = encode_categorical_features(df)
    categorical_cols = ['Country', 'Gender', 'Has_Credit_Card', 'Is_Term_Deposit']
    for col in categorical_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"'{col}' should be encoded as numeric"

def test_save_processed_data():
    df = load_data('data/bank_churn.csv')
    df = handle_missing_values(df)
    df = create_age_groups(df)
    df = encode_categorical_features(df)
    save_processed_data(df, 'data/processed_bank_churn.csv')
    processed_df = pd.read_csv('data/processed_bank_churn.csv')
    assert not processed_df.empty, "Processed data should be saved and not empty"
    assert 'Age_Group' in processed_df.columns, "'Age_Group' should exist in processed data"
