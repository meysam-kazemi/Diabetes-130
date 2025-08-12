# train.py

import pandas as pd
import logging
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Import your custom modules
from src.read_data import load_data
from src.preprocess import DataPreprocessor
from src.feature_selection import FeatureSelector

# --- 1. Configuration ---
ARTIFACTS_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FEATURES_TO_SELECT = 30
TARGET_COLUMN = 'readmitted'

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training_pipeline():
    """
    Executes the full machine learning training pipeline:
    loading, splitting, preprocessing, feature selection, resampling,
    training, evaluation, and saving artifacts.
    """
    # Create directory for saving models and other artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Artifacts will be saved in: {ARTIFACTS_DIR.resolve()}")

    # --- 2. Load Data ---
    logging.info("Loading data...")
    X, y = load_data()
    # Convert target variable to binary format
    y = y[TARGET_COLUMN].replace({'<30': 1, '>30': 0, 'NO': 0})
    X = X.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
    logging.info(f"Data loaded successfully. Features shape: {X.shape}, Target shape: {y.shape}")

    # --- 3. Split Data ---
    logging.info(f"Splitting data into training and testing sets (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- 4. Preprocess Data ---
    logging.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    logging.info("Preprocessing complete.")

    # --- 5. Select Features ---
    logging.info(f"Selecting best {N_FEATURES_TO_SELECT} features using RFE...")
    selector = FeatureSelector(n_features_to_select=N_FEATURES_TO_SELECT)
    # Align y_train index with the processed (and potentially cleaned) X_train_processed
    y_train_aligned = y_train.loc[X_train_processed.index]
    X_train_selected = selector.fit_transform(X_train_processed, y_train_aligned)
    X_test_selected = selector.transform(X_test_processed)
    logging.info(f"Feature selection complete. New shape: {X_train_selected.shape}")

    # --- 6. Handle Class Imbalance ---
    logging.info("Handling class imbalance using ADASYN...")
    adasyn = ADASYN(random_state=RANDOM_STATE)
    X_resampled, y_resampled = adasyn.fit_resample(X_train_selected, y_train_aligned.loc[X_train_selected.index])
    logging.info(f"Resampling complete. Resampled data shape: {X_resampled.shape}")

    # --- 7. Train Model ---
    logging.info("Training XGBoost model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    model.fit(X_resampled, y_resampled)
    logging.info("Model training complete.")

    # --- 8. Evaluate Model ---
    logging.info("Evaluating model on the test set...")
    # Align y_test index with the processed and selected X_test
    y_test_aligned = y_test.loc[X_test_selected.index]
    predictions = model.predict(X_test_selected)
    pred_proba = model.predict_proba(X_test_selected)[:, 1]

    report = classification_report(y_test_aligned, predictions)
    roc_auc = roc_auc_score(y_test_aligned, pred_proba)

    logging.info(f"\n--- Classification Report ---\n{report}")
    logging.info(f"--- ROC AUC Score: {roc_auc:.4f} ---")

    # --- 9. Save Artifacts ---
    logging.info("Saving training artifacts...")
    with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open(ARTIFACTS_DIR / "feature_selector.pkl", "wb") as f:
        pickle.dump(selector, f)
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    logging.info("Artifacts saved successfully.")
    logging.info("--- Pipeline Finished ---")


if __name__ == "__main__":
    run_training_pipeline()
