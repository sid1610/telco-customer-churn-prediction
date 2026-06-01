# Telco Customer Churn Prediction

This project is a Streamlit web app that predicts whether a telecom customer is likely to churn using a trained machine learning model.

## Files

- `app.py` - Streamlit application.
- `telco_churn_model.pkl` - Trained churn prediction model.
- `preprocessor.pkl` - Fitted preprocessing pipeline.
- `requirements.txt` - Python dependencies for deployment.
- `runtime.txt` - Python runtime version for Streamlit Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deploy this repository on Streamlit Cloud and set the main file path to:

```text
app.py
```
