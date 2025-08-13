# Predicting 30-Day Readmission for Diabetic Patients

This repository contains a complete, production-style machine learning pipeline to predict whether a diabetic patient will be readmitted to the hospital within 30 days. The project showcases a robust, end-to-end workflow, including advanced feature selection, cross-validation with imbalance handling, model training, and evaluation.

## 🎯 Project Goal

Hospital readmissions are a key indicator of healthcare quality and a significant driver of medical costs. This project aims to build a reliable classification model that identifies diabetic patients at high risk of early readmission. By flagging these patients, hospitals can implement targeted interventions, improve patient outcomes, and reduce financial penalties associated with high readmission rates.

## 🛠️ Tech Stack

  * **Language:** Python 3.8+
  * **Core Libraries:** Pandas, NumPy, Scikit-learn, imbalanced-learn
  * **Modeling:** XGBoost, RandomForest
  * **Data Source:** `ucimlrepo`

## 📂 Repository Structure

The project is organized into a modular structure for clarity and maintainability:

```
.
├── data/
│   └── cache/
│       └── diabetes_dataset.pkl    # Cached raw data for faster loading
├── models/
│   ├── preprocessor.pkl            # Saved data preprocessor object
│   ├── feature_selector.pkl        # Saved feature selector object
│   └── XGBoostClassifier_final.pkl # Saved final trained model
├── src/
│   ├── __init__.py
│   ├── read_data.py                # Module for loading and caching data
│   ├── preprocess.py               # OOP class for data preprocessing
│   └── feature_selection.py        # OOP class for feature selection
├── train.py                        # Main script to run the entire training pipeline
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## 🚀 Installation and Usage

Follow these steps to set up the environment and run the training pipeline.

**1. Clone the Repository**

```
git clone https://github.com/meysam-kazemi/Diabetes-130.git
cd Diabetes-130
```

**2. Create and Activate a Virtual Environment (Recommended)**

```
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**

```
pip install -r requirements.txt
```

**4. Run the Training Pipeline**
This single command will execute all steps: data loading, preprocessing, feature selection, training, evaluation, and saving the final artifacts to the `models/` directory.

```
python train.py
```

## ⚙️ The ML Pipeline Explained

The project follows a robust, multi-stage pipeline designed to ensure model reliability and prevent data leakage.

### 1\. Data Loading & Preparation

Data is fetched from the UCI repository using a custom loader in `src/read_data.py`. This module includes file-based caching to avoid re-downloading on every run. The initial train-test split (80/20) is performed at this stage to create a held-out test set that the model will not see until the final evaluation.

### 2\. Preprocessing

An OOP `DataPreprocessor` class in `src/preprocess.py` handles all cleaning and feature engineering. This includes cleaning irrelevant data, encoding categorical features, and creating new features like ICD-9 diagnosis groups and log-transformed numerical columns.

### 3\. Feature Selection with RFE

To reduce model complexity and improve performance, **Recursive Feature Elimination (RFE)** is used. An OOP `FeatureSelector` class wraps this logic.

  * **How it works:** RFE iteratively trains an estimator (in this case, a `RandomForestClassifier`) and removes the least important feature until the desired number of features (30) is reached. This ensures that only the most impactful features are used for training.

### 4\. Model Validation with K-Fold and ADASYN

To get a reliable estimate of model performance, **Stratified K-Fold Cross-Validation** (with 5 folds) is performed on the training data.

  * **Inside each fold:**
    1.  The training data for that fold is resampled using **ADASYN (Adaptive Synthetic Sampling)**. This technique intelligently creates synthetic samples for the minority class (readmitted patients), correcting the class imbalance without polluting the validation set.
    2.  The models (`XGBoost` and `RandomForest`) are trained on the resampled data.
    3.  The trained models are evaluated on the fold's validation set (which has the original, imbalanced data).
  * **Why this is important:** This process ensures that the model is always evaluated on unseen data and that the resampling only happens on the data used for training, preventing data leakage and giving a true measure of the model's generalizability.

### 5\. Final Model Training & Evaluation

After cross-validation, the best-performing model (`XGBoostClassifier`) is re-trained on the **entire resampled training dataset**. This final model is then evaluated one last time on the held-out test set created in Step 1.

### 6\. Artifact Persistence

The `preprocessor`, `feature_selector`, and the final trained `model` are saved as `.pkl` files. This allows the entire pipeline to be easily loaded and used for making predictions on new data without retraining.

--------
## ⚖️ License

This project is licensed under the Apache License 2.0.

```
Copyright 2024 [Your Name]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 💡 Answers to Key Questions

#### 1\. How would you measure model accuracy for this task?

Simple accuracy is misleading due to class imbalance. The primary metrics for this task are:

  * **Recall (Sensitivity):** This is the most critical metric in a clinical setting. It measures our ability to identify at-risk patients, minimizing false negatives.
  * **Precision:** Important for ensuring that interventions are targeted at genuinely high-risk patients, minimizing false positives.
  * **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both.

#### 2\. How would you check for bias in your model?

I would perform a **bias audit** by segmenting the test set by sensitive demographic features (e.g., `race`, `gender`, `age`) and calculating performance metrics for each subgroup. Significant disparities in metrics like the **False Negative Rate** between groups would indicate that the model is biased and requires mitigation (e.g., through re-weighting or targeted data augmentation).

#### 3\. How would you make your model’s predictions explainable to clinicians?

To build trust with clinicians, I would use **Explainable AI (XAI)** techniques. Instead of providing a "black box" prediction, I would use libraries like **SHAP (SHapley Additive exPlanations)** to generate local, patient-specific explanations. A SHAP plot can show exactly which factors (e.g., "number of prior inpatient visits," "A1C result") contributed most to a specific patient's risk score, making the prediction transparent and actionable.
