"""
Data Preprocessing Module
Handles data cleaning, text preprocessing, and dataset matching
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

def clean_inspection_data(df):
    """
    Clean restaurant inspection data

    Parameters:
    -----------
    df : pandas.DataFrame
        Raw inspection data

    Returns:
    --------
    pandas.DataFrame
        Cleaned inspection data
    """
    # Drop columns with >90% missing values
    threshold = len(df) * 0.1
    df_clean = df.dropna(axis=1, thresh=threshold).copy()

    # Handle date columns
    for col in df_clean.select_dtypes(include=['datetime64[ns]']):
        df_clean[col] = df_clean[col].mask(df_clean[col] == pd.Timestamp('1900-01-01'))

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    return df_clean.reset_index(drop=True)


def clean_text(text):
    """
    Clean and normalize text data

    Parameters:
    -----------
    text : str
        Raw text string

    Returns:
    --------
    str
        Cleaned text
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def match_inspection_to_reviews(inspection_df, review_df):
    """
    Match inspection records to review data

    Parameters:
    -----------
    inspection_df : pandas.DataFrame
        Inspection data with restaurant info
    review_df : pandas.DataFrame
        Review data with labels and text

    Returns:
    --------
    pandas.DataFrame
        Merged dataset
    """
    # Add index for matching
    inspection_df['match_id'] = range(len(inspection_df))
    review_df['match_id'] = range(len(review_df))

    # Merge on match_id
    merged = pd.merge(
        inspection_df,
        review_df,
        on='match_id',
        how='inner'
    )

    return merged


def encode_labels(df, label_column='GRADE'):
    """
    Encode categorical labels for modeling

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with categorical labels
    label_column : str
        Name of column to encode

    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded labels
    dict
        Mapping dictionary
    """
    le = LabelEncoder()
    df_encoded = df.copy()

    # Handle missing values
    df_encoded[label_column] = df_encoded[label_column].fillna('Unknown')

    # Encode
    df_encoded[f'{label_column}_encoded'] = le.fit_transform(df_encoded[label_column])

    # Create mapping
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    return df_encoded, mapping


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully")
