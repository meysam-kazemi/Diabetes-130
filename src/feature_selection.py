# src/feature_selection.py

import pandas as pd
import logging
from typing import List
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureSelector:
    """
    A wrapper for Scikit-learn's Recursive Feature Elimination (RFE)
    that fits the fit/transform API for easy pipeline integration.
    """

    def __init__(self, n_features_to_select: int = 30, estimator=None):
        """
        Initializes the FeatureSelector.

        Args:
            n_features_to_select (int): The number of top features to select.
            estimator: The supervised learning estimator to use with RFE.
                       If None, defaults to RandomForestClassifier.
        """
        if estimator is None:
            # Default estimator is a RandomForestClassifier for its robustness
            self.estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            self.estimator = estimator

        self.n_features = n_features_to_select
        self.rfe = RFE(estimator=self.estimator, n_features_to_select=self.n_features)
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the RFE model to the data to find the best features.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values.

        Returns:
            self: The fitted FeatureSelector instance.
        """
        logging.info(f"Starting feature selection for {self.n_features} features...")
        self.rfe.fit(X, y)
        
        # Get the names of the selected columns
        self.selected_features_ = X.columns[self.rfe.support_].tolist()
        
        logging.info("Feature selection complete.")
        logging.info(f"Selected features: {self.selected_features_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduces the input DataFrame to only the selected features.

        Args:
            X (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: A DataFrame with only the selected features.
        
        Raises:
            ValueError: If the selector has not been fitted yet.
        """
        if not self.selected_features_:
            raise ValueError("FeatureSelector has not been fitted yet. Call .fit() first.")
        
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        A convenience method that fits the selector and then transforms the data.

        Args:
            X (pd.DataFrame): The training input samples.
            y (pd.Series): The target values.

        Returns:
            pd.DataFrame: The transformed DataFrame with only the selected features.
        """
        self.fit(X, y)
        return self.transform(X)
