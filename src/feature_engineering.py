# src/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Load the dataset into a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """
    Handle missing values appropriately.
    """
    df = df.dropna()
    return df

def create_age_groups(df):
    """
    Create age groups from the 'Age' column.
    """
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['0-25', '26-35', '36-45', '46-55', '56-65', '66+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

def encode_categorical_features(df):
    """
    Encode categorical features using Label Encoding.
    """
    le = LabelEncoder()
    categorical_cols = ['Country', 'Gender', 'Has_Credit_Card', 'Is_Term_Deposit']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df

def save_processed_data(df, filepath):
    """
    Save the processed DataFrame to a CSV file.
    """
    df.to_csv(filepath, index=False)

def main():
    # Load data
    df = load_data('data/bank_churn.csv')
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Create age groups
    df = create_age_groups(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Save processed data
    save_processed_data(df, 'data/processed_bank_churn.csv')

if __name__ == "__main__":
    main()
