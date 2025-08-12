# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Define constants for better maintainability
MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
    'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone',
    'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide'
]

COLS_TO_DROP = ['weight', 'payer_code', 'citoglipton', 'examide']

class DataPreprocessor:
    """
    A class to preprocess the diabetes dataset. It handles data cleaning,
    feature engineering, encoding, and scaling, mimicking Scikit-learn's
    fit/transform pattern.
    """

    def __init__(self, low_variance_threshold=0.01):
        """
        Initializes the preprocessor with a scaler and feature selector.
        """
        self.scaler = StandardScaler()
        self.variance_selector = VarianceThreshold(threshold=low_variance_threshold)
        self.selected_med_cols_ = None
        self.numeric_cols_ = None
        self.trained_columns_ = None

    @staticmethod
    def _map_icd9_code(code_str):
        """Groups ICD-9 codes into broader, clinically relevant categories."""
        if pd.isna(code_str) or code_str == '?':
            return 'Missing'
        if code_str.startswith('V'):
            return 'Supplementary'
        if code_str.startswith('E'):
            return 'External_Injury'

        try:
            code = float(code_str)
            if 1 <= code <= 139: return 'Infectious_Diseases'
            elif 140 <= code <= 239: return 'Neoplasms'
            elif 240 <= code <= 279: return 'Endocrine_Metabolic'
            elif 390 <= code <= 459: return 'Circulatory_System'
            elif 460 <= code <= 519: return 'Respiratory_System'
            elif 520 <= code <= 579: return 'Digestive_System'
            else: return 'Other'
        except ValueError:
            return 'Other'

    def _clean_data(self, df, y_series=None):
        """Drops unnecessary columns and rows."""
        df = df.drop(columns=COLS_TO_DROP, errors='ignore')

        # Filter out invalid gender rows from both X and y
        if 'gender' in df.columns and y_series is not None:
            invalid_gender_mask = df['gender'] != 'Unknown/Invalid'
            df = df[invalid_gender_mask].copy()
            y_series = y_series[invalid_gender_mask].copy()

        # Filter out deceased patients
        if 'discharge_disposition_id' in df.columns and y_series is not None:
            deceased_mask = df['discharge_disposition_id'] != 11
            df = df[deceased_mask].copy()
            y_series = y_series[deceased_mask].copy()

        return df, y_series

    def _encode_features(self, df):
        """Encodes categorical and binary features into numerical format."""
        # Age encoding
        for i in range(10):
            df['age'] = df['age'].replace(f'[{10*i}-{10*(i+1)})', i + 1)

        # Medication encoding
        for col in MEDICATION_COLS:
            if col in df.columns:
                df[col] = df[col].map({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1}).fillna(0)

        # Binary feature encoding
        df['change'] = df['change'].replace({'No': 0, 'Ch': 1})
        df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0})
        df['diabetesMed'] = df['diabetesMed'].replace({'Yes': 1, 'No': 0})

        # Special result encoding
        df['max_glu_serum'] = df['max_glu_serum'].replace({'>200': 1, '>300': 1, 'Norm': 0, 'None': -1})
        df['A1Cresult'] = df['A1Cresult'].replace({'>7': 1, '>8': 1, 'Norm': 0, 'None': -1})
        return df

    def _create_features(self, df):
        """Engineers new features from existing data."""
        # ICD-9 Grouping
        for col in ['diag_1', 'diag_2', 'diag_3']:
            df[f'{col}_group'] = df[col].apply(self._map_icd9_code)
        df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])

        # Log transformations for skewed numerical features
        for col in ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient']:
            df[col + '_log'] = np.log1p(df[col])

        # Ratio and interaction features
        df['meds_per_diag'] = df['num_medications'] / df['number_diagnoses'].replace(0, 1)
        df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        
        df = pd.get_dummies(df, columns=['medical_specialty'], drop_first=True)
        return df

    def fit(self, X, y):
        """
        Fits the preprocessor on the training data. Learns scaling parameters,
        feature selection, and final column structure.
        
        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.
        
        Returns:
            pd.DataFrame: The transformed training data.
        """
        X, y = self._clean_data(X.copy(), y.copy())
        X = self._encode_features(X)

        # Select medication columns based on variance
        med_cols_in_data = [col for col in MEDICATION_COLS if col in X.columns]
        self.variance_selector.fit(X[med_cols_in_data])
        self.selected_med_cols_ = X[med_cols_in_data].columns[self.variance_selector.get_support()].tolist()

        # Keep only selected medication columns
        cols_to_keep = [col for col in X.columns if col not in med_cols_in_data] + self.selected_med_cols_
        X = X[cols_to_keep]

        X = self._create_features(X)

        # One-hot encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Identify numeric columns for scaling and fit the scaler
        self.numeric_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        X[self.numeric_cols_] = self.scaler.fit_transform(X[self.numeric_cols_])

        # Feature set
        feature_set = self.numeric_cols_ + [col for col in X.columns if col.startswith((
            'medical_specialty', 'max_glu_serum_', 'A1Cresult_', 'diag_1_group_', 'diag_2_group_', 'diag_3_group_'
        ))] + MEDICATION_COLS + ['gender', 'change', 'diabetesMed']
        X = X[feature_set]
        # Store the final columns to ensure test set consistency
        self.trained_columns_ = X.columns
        return X

    def transform(self, X):
        """
        Transforms new data using the already fitted preprocessor.
        
        Args:
            X (pd.DataFrame): The data to transform (e.g., test set).
        
        Returns:
            pd.DataFrame: The transformed data.
        """
        X_copy = X.copy()
        X_copy, _ = self._clean_data(X_copy)
        X_copy = self._encode_features(X_copy)

        # Keep only the selected medication columns from training
        med_cols_in_data = [col for col in MEDICATION_COLS if col in X_copy.columns]
        cols_to_keep = [col for col in X_copy.columns if col not in med_cols_in_data] + self.selected_med_cols_
        X_copy = X_copy[cols_to_keep]

        X_copy = self._create_features(X_copy)
        
        # One-hot encode using the same logic
        categorical_cols = X_copy.select_dtypes(include=['object', 'category']).columns
        X_copy = pd.get_dummies(X_copy, columns=categorical_cols, drop_first=True)

        # Align columns with the training set
        missing_cols = set(self.trained_columns_) - set(X_copy.columns)
        for c in missing_cols:
            X_copy[c] = 0
        X_copy = X_copy[self.trained_columns_] # Ensure same order and columns

        # Transform numeric columns with the fitted scaler
        X_copy[self.numeric_cols_] = self.scaler.transform(X_copy[self.numeric_cols_])

        return X_copy
