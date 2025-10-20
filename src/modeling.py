"""
Machine Learning Model Training Module
Implements Logistic Regression, SVM, and Random Forest classifiers
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    """
    Class to train and manage multiple classification models
    """

    def __init__(self, random_state=42):
        """
        Initialize model trainer

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.vectorizer = None
        self.scaler = None

    def prepare_features(self, X_text, X_numeric=None):
        """
        Prepare text and numeric features

        Parameters:
        -----------
        X_text : array-like
            Text data for vectorization
        X_numeric : array-like, optional
            Additional numeric features

        Returns:
        --------
        array
            Combined feature matrix
        """
        # Vectorize text
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_text_vec = self.vectorizer.fit_transform(X_text)
        else:
            X_text_vec = self.vectorizer.transform(X_text)

        # Combine with numeric features if provided
        if X_numeric is not None:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            else:
                X_numeric_scaled = self.scaler.transform(X_numeric)

            X_combined = np.hstack([X_text_vec.toarray(), X_numeric_scaled])
        else:
            X_combined = X_text_vec.toarray()

        return X_combined

    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels

        Returns:
        --------
        LogisticRegression
            Trained model
        """
        lr = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr

        return lr

    def train_svm(self, X_train, y_train):
        """
        Train Support Vector Machine model

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels

        Returns:
        --------
        SVC
            Trained model
        """
        svm = SVC(
            kernel='rbf',
            random_state=self.random_state,
            class_weight='balanced',
            probability=True
        )
        svm.fit(X_train, y_train)
        self.models['svm'] = svm

        return svm

    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels

        Returns:
        --------
        RandomForestClassifier
            Trained model
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced',
            max_depth=10
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        return rf

    def train_all_models(self, X_train, y_train):
        """
        Train all models

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels

        Returns:
        --------
        dict
            Dictionary of trained models
        """
        print("Training Logistic Regression...")
        self.train_logistic_regression(X_train, y_train)

        print("Training SVM...")
        self.train_svm(X_train, y_train)

        print("Training Random Forest...")
        self.train_random_forest(X_train, y_train)

        print("All models trained successfully!")

        return self.models


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets

    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Proportion of test data
    random_state : int
        Random seed

    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    print("Modeling module loaded successfully")
