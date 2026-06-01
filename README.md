# Telco Customer Churn Intelligence

An industry-style Streamlit application for predicting telecom customer churn from trained machine learning artifacts. The app supports single-customer scoring, batch CSV scoring, risk banding, and model/schema inspection.

## Features

- Production-oriented Streamlit layout with separate workflows.
- Cached model and preprocessor loading.
- Explicit input schema and validation.
- Single customer churn prediction with recommended retention action.
- Batch CSV scoring with downloadable predictions.
- Deployment notes for Streamlit Cloud.

## Project Structure

```text
.
├── app.py
├── src/
│   ├── config.py
│   └── model_service.py
├── data/
│   └── sample_batch.csv
├── docs/
│   └── DEPLOYMENT.md
├── telco_churn_model.pkl
├── preprocessor.pkl
├── requirements.txt
└── runtime.txt
```

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

The model was trained with `scikit-learn==1.5.1`, so the dependency is pinned in `requirements.txt`.
