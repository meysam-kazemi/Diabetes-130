# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import VarianceThreshold, RFE
import warnings
warnings.filterwarnings('ignore')

def map_icd9_code(code_str):
    """Groups ICD-9 codes into broader categories."""
    if pd.isna(code_str) or code_str == '?':
        return 'Missing'

    # Handle codes that start with letters (V-codes, E-codes)
    if code_str.startswith('V'):
        return 'Supplementary'
    if code_str.startswith('E'):
        return 'External Injury'

    try:
        # Convert to float for numeric comparison
        code = float(code_str)

        # Define ranges for grouping
        if 1 <= code <= 139:
            return 'Infectious Diseases'
        elif 140 <= code <= 239:
            return 'Neoplasms' # Cancer
        elif 240 <= code <= 279:
            # This group includes Diabetes (250.xx)
            return 'Endocrine/Metabolic'
        elif 390 <= code <= 459:
            return 'Circulatory System'
        elif 460 <= code <= 519:
            return 'Respiratory System'
        elif 520 <= code <= 579:
            return 'Digestive System'
        # Add more 'elif' blocks here for other ranges if needed
        else:
            return 'Other'

    except ValueError:
        # If conversion to float fails for any other reason
        return 'Other'



# Encode age
ages = X.age.unique()
print("Ages: {}".format(ages))
for i in range(10):
    X.age = X.age.replace(f'[{10*i}-{10*(i+1)})', i+1)




medication_cols = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
    'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone',
    'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide'
]

for c in medication_cols:
    print(f"{c:<25}: {X[c].unique()}")


# Encode medication columns as numeric (0 for "No", 1 for "Steady"/"Up"/"Down")
for col in medication_cols:
    X[col] = X[col].map({'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1})

X['change'] = X['change'].replace({'No': 0, 'Ch': 1})
y['readmitted'] = y['readmitted'].replace({'<30': 1, '>30': 0, 'NO': 0})
X['gender'] = X['gender'].replace({'Male': 1, 'Female': 0})
X['diabetesMed'] = X['diabetesMed'].replace({'Yes': 1, 'No': 0})
X['max_glu_serum'] = X['max_glu_serum'].replace({'>200': 1, '>300': 1, 'Norm': 0, 'Not Tested': -1})
X['A1Cresult'] = X['A1Cresult'].replace({'>7': 1, '>8': 1, 'Norm': 0, 'Not Tested': -1})
X = X.fillna('Unknown')

# drop miss rows
X = X.drop(['weight', 'payer_code', 'citoglipton', 'examide'], axis=1)
y = y[X['gender'] != 'Unknown/Invalid']
X = X[X['gender'] != 'Unknown/Invalid']


# Drop deceased patients
y = y[X['discharge_disposition_id'] != 11]
X = X[X['discharge_disposition_id'] != 11]
X.shape, y.shape

# Drop low-variance medication columns
selector = VarianceThreshold(threshold=0.01)
selector.fit(X[medication_cols])
medication_cols = [medication_cols[i] for i in range(len(medication_cols)) if selector.get_support()[i]]


X['diag_1_group'] = X['diag_1'].apply(map_icd9_code)
X['diag_2_group'] = X['diag_2'].apply(map_icd9_code)
X['diag_3_group'] = X['diag_3'].apply(map_icd9_code)
X[['diag_1', 'diag_2', 'diag_3', 'diag_1_group', 'diag_2_group', 'diag_3_group']]



for col in ['time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_inpatient']:
    X[col + '_log'] = np.log1p(X[col])

# Create ratio features
X['meds_per_diag'] = X['num_medications'] / X['number_diagnoses'].replace(0, 1)
X['hospital_per_age'] = X['time_in_hospital'] / X['age'].replace(0, 1)



# One-hot encode categorical variables
categorical_cols = ['max_glu_serum', 'A1Cresult', 'diag_1_group', 'diag_2_group', 'diag_3_group']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Interaction terms
interaction_terms = [('num_medications', 'time_in_hospital'), ('num_medications', 'number_diagnoses')]
for inter in interaction_terms:
    name = inter[0] + '|' + inter[1]
    X[name] = X[inter[0]] * X[inter[1]]

# Define numeric columns
numeric_cols = [
    'age', 'time_in_hospital_log', 'num_lab_procedures_log', 'num_procedures', 'num_medications_log',
    'number_outpatient', 'number_emergency', 'number_inpatient_log', 'number_diagnoses',
    'service_utilization_log', 'numchange', 'encounter_count', 'meds_per_diag', 'hospital_per_age',
    'num_medications|time_in_hospital', 'num_medications|number_diagnoses'
]



# Align X and y based on their index
y = y.loc[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()


X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# Identify remaining non-numeric columns
categorical_cols_remaining = X_train.select_dtypes(include=['object', 'category']).columns

# One-hot encode the remaining categorical columns
X_train = pd.get_dummies(X_train, columns=categorical_cols_remaining, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols_remaining, drop_first=True)

# Align columns after one-hot encoding - critical for consistent feature sets
X_train, X_test = X_train.align(X_test, join='inner', axis=1, fill_value=0)

# Instantiate ADASYN
adasyn = ADASYN(random_state=42)

# Apply ADASYN to the training data
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

display(X_resampled.head())
display(y_resampled.head())
display(y_resampled.value_counts())


models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LGBMClassifier': LGBMClassifier(random_state=42)
}


