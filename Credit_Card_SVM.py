# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load datasets
df1 = pd.read_csv('/content/drive/MyDrive/datasets/credit_record.csv')
df2 = pd.read_csv('/content/drive/MyDrive/datasets/application_record.csv')
df = pd.merge(df1, df2, on='ID')

# Clean and map categorical values
df['STATUS'] = df['STATUS'].map({'0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, 'C': 0, 'X': 0})
df['CODE_GENDER'] = df['CODE_GENDER'].map({'M': 1, 'F': 0})
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
df.drop(['ID', 'MONTHS_BALANCE'], axis=1, inplace=True)

# Fill null values
df["OCCUPATION_TYPE"].fillna(df["OCCUPATION_TYPE"].mode()[0], inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", 
                                 "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", 
                                 "OCCUPATION_TYPE"], drop_first=True)

# Convert all boolean to integers
df = df.astype(int)

# Convert days to years
df['YEARS_BIRTH'] = abs(df['DAYS_BIRTH']) / 365
df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365
df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"], inplace=True)

# Min-max scaling
scaler = MinMaxScaler()
scale_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'YEARS_BIRTH', 'YEARS_EMPLOYED', 'CNT_FAM_MEMBERS']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Split features and labels
X = df.drop('STATUS', axis=1)
y = df['STATUS']

# Feature selection
selector = SelectKBest(score_func=f_classif, k=20)
X_new = selector.fit_transform(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Train SVM model
svm_model = SVC(kernel="linear", C=1.0, class_weight='balanced', verbose=True)
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()
