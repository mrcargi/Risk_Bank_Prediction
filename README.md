# Risk Prediction System API

This project is a Machine Learning-based API for predicting credit risk. The API allows users to submit loan applicant data and receive a prediction of whether the applicant is a good or bad credit risk.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mrcargi/risk-prediction-system.git
   cd risk-prediction-system

2. Create and activate a virtual enviroment     
    source venv/bin/activate
3. pip install -r requirements.txt




## How to use 

### Endpoint 1: get data to predict 


- **URL:** `/predict/{model_name}`
- **Method:** `POST`
- **Description:** Submit loan applicant data to receive a credit risk prediction. Choose the model you want to use from `"svm"`, `"naive_bayes"`, `"random_forest"`. You need a specific nomenclature for some parameters, which you can find here: [Credit Data Page](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). Make sure that you have checked the parameters before making the request.


### Examples: 
    ```bash
curl -X POST "http://localhost:8000/predict/{model_name}/" \
-H "Content-Type: application/json" \
-d '{
  "CheckingAccountStatus": "A11",
  "DurationInMonths": 6,
  "CreditHistory": "A34",
  "Purpose": "A43",
  "CreditAmount": 1169,
  "SavingsAccountBonds": "A65",
  "Employment": "A75",
  "InstallmentRatePercentage": 4,
  "PersonalStatusSex": "A93",
  "OtherDebtorsGuarantors": "A101",
  "ResidenceSince": 4,
  "Property": "A121",
  "Age": 67,
  "OtherInstallmentPlans": "A143",
  "Housing": "A152",
  "NumberOfExistingCredits": 2,
  "Job": "A173",
  "PeopleUnderMaintenance": 1,
  "Telephone": "A191",
  "ForeignWorker": "A201"
}'