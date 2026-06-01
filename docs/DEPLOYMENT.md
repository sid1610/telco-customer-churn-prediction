# Deployment Guide

## Streamlit Cloud

1. Connect the GitHub repository.
2. Select the `main` branch.
3. Set the app entry point to `app.py`.
4. Deploy.

## Required Artifacts

The app expects these files in the repository root:

- `telco_churn_model.pkl`
- `preprocessor.pkl`

## Runtime

The model was trained with `scikit-learn==1.5.1`, so the dependency is pinned in `requirements.txt`.

## Smoke Test

After deployment, open the app and confirm:

- The sidebar shows "Model loaded".
- Single prediction returns a churn result.
- Batch scoring accepts a CSV with the required schema.
