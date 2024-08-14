# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:27:37 2024

@author: GIOVANI
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

column_names = [
    'CheckingAccountStatus', 'DurationInMonths', 'CreditHistory', 'Purpose', 'CreditAmount',
    'SavingsAccountBonds', 'Employment', 'InstallmentRatePercentage', 'PersonalStatusSex',
    'OtherDebtorsGuarantors', 'ResidenceSince', 'Property', 'Age', 'OtherInstallmentPlans',
    'Housing', 'NumberOfExistingCredits', 'Job', 'PeopleUnderMaintenance', 'Telephone', 'ForeignWorker', 'CreditRisk'
]

data = pd.read_csv('german.data', sep=' ', header=None, names=column_names, na_values='?')
data.dropna(inplace=True)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

y = y - 1  # Ajustar etiquetas a {0, 1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

categorical_features = [
    'CheckingAccountStatus', 'CreditHistory', 'Purpose', 'SavingsAccountBonds', 'Employment',
    'PersonalStatusSex', 'OtherDebtorsGuarantors', 'Property', 'OtherInstallmentPlans',
    'Housing', 'Job', 'Telephone', 'ForeignWorker'
]
numeric_features = [
    'DurationInMonths', 'CreditAmount', 'InstallmentRatePercentage', 'ResidenceSince', 'Age', 'NumberOfExistingCredits', 'PeopleUnderMaintenance'
]

# Crear el transformador de columnas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Ajustar y transformar los datos de entrenamiento
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Modelos a utilizar: Random Forest, SVM, Naive Bayes
models = [
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', SVC(kernel='rbf', probability=True, random_state=0)),
    ('Naive Bayes', GaussianNB())
]

results = {}

# Entrenamiento y evaluación de cada modelo
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

# Visualización de resultados
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")

# Gráfica ROC
plt.figure()
for name, model in models:
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='best')
plt.show()
