import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from joblib import dump
import os 
column_names = [
    'CheckingAccountStatus', 'DurationInMonths', 'CreditHistory', 'Purpose', 'CreditAmount',
    'SavingsAccountBonds', 'Employment', 'InstallmentRatePercentage', 'PersonalStatusSex',
    'OtherDebtorsGuarantors', 'ResidenceSince', 'Property', 'Age', 'OtherInstallmentPlans',
    'Housing', 'NumberOfExistingCredits', 'Job', 'PeopleUnderMaintenance', 'Telephone', 'ForeignWorker', 'CreditRisk'
]

data = pd.read_csv('data/german.data', sep=' ', header=None, names=column_names, na_values='?')


data.dropna(inplace = True)


X = data.iloc[:, :-1]
y = data.iloc[:,-1]

y = y - 1 

X_train ,X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2 , random_state=0)


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

# Transformar los datos de prueba
X_test = preprocessor.transform(X_test)



models  = [('random_forest',RandomForestClassifier()),
           ('svm',SVC(kernel='rbf',random_state=0)),
           ('naive_bayes',GaussianNB())]


for model_name, model in models:
    model_dir = f'app/models/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    model.fit(X_train, y_train)
    dump(model,  f'app/models/{model_name}/model.pkl')
    dump(preprocessor, f'app/models/{model_name}/preprocessor.pkl')
    
    
#Risk prediction sistem