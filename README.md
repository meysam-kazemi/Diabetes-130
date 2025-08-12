# Predicting 30-Day Readmission for Diabetic Patients

This repository contains a complete machine learning pipeline to predict whether a diabetic patient will be readmitted to the hospital within 30 days. The project demonstrates a robust, end-to-end workflow, including data preprocessing, feature engineering, model training, evaluation, and artifact persistence.

## 🎯 Project Goal

Hospital readmissions are a key indicator of healthcare quality and a significant driver of medical costs. This project aims to build a reliable classification model that identifies diabetic patients at high risk of early readmission. By flagging these patients, hospitals can implement targeted interventions, improve patient outcomes, and reduce financial penalties associated with high readmission rates.

## 🛠️ Tech Stack

  * **Language:** Python 3.8+
  * **Core Libraries:** Pandas, NumPy, Scikit-learn, imbalanced-learn
  * **Modeling:** XGBoost, RandomForestClassifier 
  * **Data Source:** `ucimlrepo`

## 📂 Repository Structure

The project is organized into a modular structure for clarity and maintainability:

```
.
├── data/
│   └── cache/
│       └── diabetes_dataset.pkl   
├── models/
│   ├── preprocessor.pkl          
│   ├── feature_selector.pkl     
│   └── model.pkl               
├── src/
│   ├── __init__.py
│   ├── read_data.py           
│   ├── preprocess.py         
│   └── feature_selection.py 
├── train.py                
├── requirements.txt       
└── README.md             
```

## 🚀 Installation and Usage

Follow these steps to set up the environment and run the training pipeline.

**1. Clone the Repository**

```bash
git clone https://github.com/meysam-kazemi/Diabetes-130.git
cd Diabetes-130
```

**2. Create and Activate a Virtual Environment (Recommended)**

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Training Pipeline**
This single command will execute all steps: data loading, preprocessing, feature selection, training, evaluation, and saving the final artifacts to the `models/` directory.

```bash
python train.py
```

## ⚙️ The ML Pipeline

The project follows a structured machine learning pipeline:

1.  **Data Loading:** Data is fetched from the UCI repository using a custom loader in `src/read_data.py` that includes caching for faster subsequent runs.
2.  **Preprocessing:** An OOP `DataPreprocessor` class in `src/preprocess.py` handles all cleaning and feature engineering:
      * **Cleaning:** Removes irrelevant columns and rows corresponding to deceased patients.
      * **Encoding:** Converts categorical features (`age`, `gender`, medication status) into a numerical format.
      * **Feature Engineering:**
          * **ICD-9 Grouping:** Maps over 14,000 unique `ICD-9` diagnosis codes into 9 clinically relevant categories to reduce dimensionality.
          * **Log Transformation:** Applies `log1p` to skewed numerical features like `time_in_hospital`.
          * **Ratio Features:** Creates new features like `meds_per_diag`.
3.  **Feature Selection:** An OOP `FeatureSelector` class in `src/feature_selection.py` uses **Recursive Feature Elimination (RFE)** with a `RandomForestClassifier` to select the top 30 most impactful features.
4.  **Handling Class Imbalance:** The minority class (readmitted \<30 days) is oversampled using **ADASYN** on the training data to prevent model bias.
5.  **Model Training:** An **XGBoost Classifier** is trained on the preprocessed, selected, and resampled data.
6.  **Evaluation & Artifact Persistence:** The trained model is evaluated on the unseen test set. The `preprocessor`, `feature_selector`, and `model` objects are saved as `.pkl` files for future use.

## 📊 Results

The model's performance on the held-out test set is summarized below. The focus was on achieving a good balance between **Recall** (correctly identifying at-risk patients) and **Precision**.

| Metric                | Score |
| --------------------- | ----- |
| **Recall (Class 1)** | 0.XX  |
| **Precision (Class 1)**| 0.XX  |
| **F1-Score (Class 1)** | 0.XX  |

*(Note: Replace `0.XX` with your final scores after running `train.py`)*

