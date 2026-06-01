import pandas as pd


APP_TITLE = "Telco Customer Churn Intelligence"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

YES_NO_OPTIONS = ["No", "Yes"]
MULTIPLE_LINES_OPTIONS = ["No", "No phone service", "Yes"]
INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"]
SERVICE_OPTIONS = ["No", "No internet service", "Yes"]
CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHOD_OPTIONS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65

RECOMMENDED_ACTIONS = {
    "Low": "Maintain standard engagement and continue monitoring customer experience.",
    "Medium": "Prioritize retention outreach and review billing, support, or service concerns.",
    "High": "Escalate to a retention specialist with a targeted offer or service recovery action.",
}

BATCH_TEMPLATE = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0,
        }
    ],
    columns=FEATURE_COLUMNS,
)
