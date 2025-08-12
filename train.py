# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from src.preprocess import DataPreprocessor 
from src.read_data import load_data

X, y = read_data()
y = y.replace({'<30': 1, '>30': 0, 'NO': 0})


# 3. Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 4. Preprocess Data using the OOP class
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)


# 5. Handle Class Imbalance (on training data only)
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train_processed, y_train.loc[X_train_processed.index])


# 6. Train the model
print("Training the model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_resampled, y_resampled)


# 7. Evaluate the model (example)
predictions = model.predict(X_test_processed)
print("\n--- Classification Report on Test Set ---")
print(classification_report(y_test.loc[X_test_processed.index], predictions))

print("\nPreprocessing and training complete!")
print("Shape of resampled training data:", X_resampled.shape)
print("Shape of test data:", X_test_processed.shape)
