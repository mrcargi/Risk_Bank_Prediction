from pydantic import BaseModel

class PredictionRequest(BaseModel):
    CheckingAccountStatus: str
    DurationInMonths: int
    CreditHistory: str
    Purpose: str
    CreditAmount: int
    SavingsAccountBonds: str
    Employment: str
    InstallmentRatePercentage: int
    PersonalStatusSex: str
    OtherDebtorsGuarantors: str
    ResidenceSince: int
    Property: str
    Age: int
    OtherInstallmentPlans: str
    Housing: str
    NumberOfExistingCredits: int
    Job: str
    PeopleUnderMaintenance: int
    Telephone: str
    ForeignWorker: str   