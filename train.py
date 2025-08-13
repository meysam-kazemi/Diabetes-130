# train.py

import logging
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# --- 1. Configuration ---
ARTIFACTS_DIR = Path("models")
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FEATURES_TO_SELECT = 30
TARGET_COLUMN = 'readmitted'

# --- Configure Logging ---
logging.basicConfig(level=print, format='%(asctime)s - %(levelname)s - %(message)s')

models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
}

# def run_training_pipeline():
# Create directory for saving models and other artifacts
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Artifacts will be saved in: {ARTIFACTS_DIR.resolve()}")

# --- 2. Load Data ---
print("Loading data...")
X, y = load_data()
# Convert target variable to binary format
y = y[TARGET_COLUMN].replace({'<30': 1, '>30': 0, 'NO': 0})
X = X.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')
print(f"Data loaded successfully. Features shape: {X.shape}, Target shape: {y.shape}")

# --- 3. Split Data ---
print(f"Splitting data into training and testing sets (test_size={TEST_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# --- 4. Preprocess Data ---
print("Preprocessing data...")
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)
print("Preprocessing complete.")

# --- 5. Select Features ---
print(f"Selecting best {N_FEATURES_TO_SELECT} features using RFE...")
selector = FeatureSelector(n_features_to_select=N_FEATURES_TO_SELECT)
# Align y_train index with the processed (and potentially cleaned) X_train_processed
y_train_aligned = y_train.loc[X_train_processed.index]
X_train_selected = selector.fit_transform(X_train_processed, y_train_aligned)
X_test_selected = selector.transform(X_test_processed)
print(f"Feature selection complete. New shape: {X_train_selected.shape}")


# --- 6. Handle Class Imbalance and Train Model with K-Fold ---
print("Handling class imbalance and training models with K-Fold cross-validation...")

# Define the number of folds
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

# Initialize dictionaries to store results
results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for name in models.keys()}

for fold, (train_index, val_index) in enumerate(skf.split(X_train_selected, y_train_aligned)):
    print(f"--- Fold {fold + 1}/{n_splits} ---")

    X_train_fold, X_val_fold = X_train_selected.iloc[train_index], X_train_selected.iloc[val_index]
    y_train_fold, y_val_fold = y_train_aligned.iloc[train_index], y_train_aligned.iloc[val_index]

    # Handle Class Imbalance on the training fold
    adasyn = ADASYN(random_state=RANDOM_STATE)
    X_resampled_fold, y_resampled_fold = adasyn.fit_resample(X_train_fold, y_train_fold)
    print(f"Fold {fold + 1} - Resampling complete. Resampled data shape: {X_resampled_fold.shape}")

    for name, model in models.items():
        print(f"Training {name} for Fold {fold + 1}...")
        model.fit(X_resampled_fold.values, y_resampled_fold.values.ravel())
        print(f"Finished training {name} for Fold {fold + 1}.")

        # Evaluate on the validation fold
        predictions_fold = model.predict(X_val_fold.values)

        # Calculate metrics for the fold
        results[name]['accuracy'].append(accuracy_score(y_val_fold, predictions_fold))
        results[name]['precision'].append(precision_score(y_val_fold, predictions_fold))
        results[name]['recall'].append(recall_score(y_val_fold, predictions_fold))
        results[name]['f1'].append(f1_score(y_val_fold, predictions_fold))

# --- 7. Aggregate and Report Results ---
print("\n--- Aggregated Cross-Validation Results ---")
for name, metrics in results.items():
    print(f"--- {name} ---")
    print(f"Average Accuracy  : {np.mean(metrics['accuracy'])*100:.2f}%")
    print(f"Average Precision : {np.mean(metrics['precision'])*100:.2f}%")
    print(f"Average Recall    : {np.mean(metrics['recall'])*100:.2f}%")
    print(f"Average F1 Score  : {np.mean(metrics['f1'])*100:.2f}%")
    print("-" * 30)

# --- 8. Train Final Model on Full Resampled Training Data ---
print("Training final models on full resampled training data...")

adasyn_final = ADASYN(random_state=RANDOM_STATE)
X_resampled_final, y_resampled_final = adasyn_final.fit_resample(X_train_selected, y_train_aligned)
print(f"Final resampling complete. Resampled data shape: {X_resampled_final.shape}")

final_models = {}
for name, model in models.items():
    print(f"Training final {name} model...")
    final_models[name] = model # Use a fresh instance or a copy if models are stateful
    final_models[name].fit(X_resampled_final.values, y_resampled_final.values.ravel())
    print(f"Finished training final {name} model.")

# --- 9. Evaluate Final Models on Test Set ---
print("\n--- Final Model Evaluation on Test Set ---")
# Ensure X_test_selected and y_test_aligned are correctly aligned (already handled earlier)
y_test_aligned = y_test.loc[X_test_selected.index]
for name, model in final_models.items():
    print(f"Evaluating final {name} on test set...")

    # Make predictions
    predictions_test = model.predict(X_test_selected.values)

    # Calculate metrics
    acc_test = accuracy_score(y_test_aligned, predictions_test)
    f1_test = f1_score(y_test_aligned, predictions_test)
    precision_test = precision_score(y_test_aligned, predictions_test)
    recall_test = recall_score(y_test_aligned, predictions_test)

    report_test = f"Accuracy  : {acc_test*100:.2f}%\nF1        : {f1_test*100:.2f}%\nPrecision : {precision_test*100:.2f}%\nRecall    : {recall_test*100:.2f}%"

    print(f"\nClassification Report for {name} on Test Set:\n{report_test}")
    print("-" * 30)

# --- 10. Save Artifacts ---
print("Saving training artifacts...")
with open(ARTIFACTS_DIR / "preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
with open(ARTIFACTS_DIR / "feature_selector.", "wb") as f:
    pickle.dump(selector, f)
for name, model in final_models.items():
    with open(ARTIFACTS_DIR / f"{name}_final.pkl", "wb") as f:
        pickle.dump(model, f)
print("Artifacts saved successfully.")
print("--- Pipeline Finished ---")
