from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    APP_TITLE,
    BATCH_TEMPLATE,
    CONTRACT_OPTIONS,
    FEATURE_COLUMNS,
    INTERNET_SERVICE_OPTIONS,
    MULTIPLE_LINES_OPTIONS,
    PAYMENT_METHOD_OPTIONS,
    SERVICE_OPTIONS,
    YES_NO_OPTIONS,
)
from src.model_service import (
    ArtifactLoadError,
    build_customer_frame,
    expected_model_columns,
    load_artifacts,
    score_batch,
    score_customer,
    validate_customer,
)


BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts():
    return load_artifacts(
        model_path=BASE_DIR / "telco_churn_model.pkl",
        preprocessor_path=BASE_DIR / "preprocessor.pkl",
    )


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stForm"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
        }
        .risk-high {
            color: #991b1b;
            background: #fee2e2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 1rem;
        }
        .risk-medium {
            color: #92400e;
            background: #fef3c7;
            border: 1px solid #fde68a;
            border-radius: 8px;
            padding: 1rem;
        }
        .risk-low {
            color: #166534;
            background: #dcfce7;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def selectbox(label, options, default):
    return st.selectbox(label, options, index=options.index(default))


def render_header():
    st.title(APP_TITLE)
    st.caption(
        "A production-ready Streamlit app for customer retention teams to score churn risk from trained ML artifacts."
    )


def render_sidebar(artifacts):
    with st.sidebar:
        st.header("Application")
        page = st.radio(
            "Workflow",
            ["Single prediction", "Batch scoring", "Model details"],
            label_visibility="collapsed",
        )

        st.divider()
        st.subheader("Artifact health")
        st.success("Model loaded")
        st.caption(f"Model: `{artifacts.model_path.name}`")
        st.caption(f"Preprocessor: `{artifacts.preprocessor_path.name}`")

        model_name = artifacts.model.__class__.__name__
        preprocessor_name = artifacts.preprocessor.__class__.__name__
        st.caption(f"Estimator: `{model_name}`")
        st.caption(f"Preprocessor type: `{preprocessor_name}`")

        st.divider()
        st.subheader("Risk bands")
        st.caption("Low: below 35%")
        st.caption("Medium: 35% to 65%")
        st.caption("High: 65% and above")

    return page


def render_overview():
    col1, col2, col3 = st.columns(3)
    col1.metric("Input features", len(FEATURE_COLUMNS))
    col2.metric("Supported workflows", "2")
    col3.metric("Deployment target", "Streamlit Cloud")


def render_single_prediction(artifacts):
    st.subheader("Single Customer Scoring")

    with st.form("single_prediction_form"):
        customer_col, service_col, billing_col = st.columns(3)

        with customer_col:
            st.markdown("**Customer profile**")
            gender = selectbox("Gender", ["Female", "Male"], "Female")
            senior_citizen = st.checkbox("Senior Citizen")
            partner = selectbox("Partner", YES_NO_OPTIONS, "No")
            dependents = selectbox("Dependents", YES_NO_OPTIONS, "No")
            tenure = st.number_input(
                "Tenure in months",
                min_value=0,
                max_value=100,
                value=12,
                step=1,
            )

        with service_col:
            st.markdown("**Services**")
            phone_service = selectbox("Phone Service", YES_NO_OPTIONS, "Yes")
            multiple_lines = selectbox("Multiple Lines", MULTIPLE_LINES_OPTIONS, "No")
            internet_service = selectbox(
                "Internet Service",
                INTERNET_SERVICE_OPTIONS,
                "Fiber optic",
            )

            service_default = "No internet service" if internet_service == "No" else "No"
            online_security = selectbox("Online Security", SERVICE_OPTIONS, service_default)
            online_backup = selectbox("Online Backup", SERVICE_OPTIONS, service_default)
            device_protection = selectbox(
                "Device Protection",
                SERVICE_OPTIONS,
                service_default,
            )
            tech_support = selectbox("Tech Support", SERVICE_OPTIONS, service_default)
            streaming_tv = selectbox("Streaming TV", SERVICE_OPTIONS, service_default)
            streaming_movies = selectbox("Streaming Movies", SERVICE_OPTIONS, service_default)

        with billing_col:
            st.markdown("**Billing**")
            contract = selectbox("Contract", CONTRACT_OPTIONS, "Month-to-month")
            paperless_billing = selectbox("Paperless Billing", YES_NO_OPTIONS, "Yes")
            payment_method = selectbox(
                "Payment Method",
                PAYMENT_METHOD_OPTIONS,
                "Electronic check",
            )
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0,
                step=1.0,
            )
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0,
            )

        submitted = st.form_submit_button("Score customer", use_container_width=True)

    if not submitted:
        st.info("Submit the form to generate a churn prediction.")
        return

    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    errors = validate_customer(customer)
    if errors:
        st.error("Please fix the validation issues before scoring.")
        for error in errors:
            st.caption(error)
        return

    customer_frame = build_customer_frame(customer)
    result = score_customer(artifacts, customer_frame)

    score_col, probability_col, action_col = st.columns([1.1, 1, 1.4])
    with score_col:
        if result.prediction_label == "Yes":
            st.error("Prediction: likely to churn")
        else:
            st.success("Prediction: unlikely to churn")

    with probability_col:
        if result.churn_probability is not None:
            st.metric("Churn probability", f"{result.churn_probability * 100:.1f}%")
        else:
            st.metric("Predicted class", result.prediction_label)

    with action_col:
        css_class = f"risk-{result.risk_level.lower()}"
        st.markdown(
            f"""
            <div class="{css_class}">
                <strong>{result.risk_level} risk</strong><br>
                {result.recommended_action}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Audit input sent to the model"):
        st.dataframe(customer_frame, use_container_width=True, hide_index=True)


def render_batch_scoring(artifacts):
    st.subheader("Batch CSV Scoring")
    st.write(
        "Upload a CSV with the required Telco customer columns. The app returns prediction, churn probability, and risk band for each row."
    )

    template_csv = BATCH_TEMPLATE.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV template",
        data=template_csv,
        file_name="telco_churn_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        batch = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error("Could not read the uploaded CSV.")
        st.caption(str(error))
        return

    missing_columns = [column for column in FEATURE_COLUMNS if column not in batch.columns]
    if missing_columns:
        st.error("The uploaded file is missing required columns.")
        st.caption(", ".join(missing_columns))
        return

    try:
        scored = score_batch(artifacts, batch[FEATURE_COLUMNS].copy())
    except Exception as error:
        st.error("Batch scoring failed.")
        st.caption(str(error))
        return

    high_risk_count = int((scored["risk_level"] == "High").sum())
    avg_probability = scored["churn_probability"].dropna().mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows scored", len(scored))
    col2.metric("High-risk customers", high_risk_count)
    if pd.isna(avg_probability):
        col3.metric("Average churn probability", "N/A")
    else:
        col3.metric("Average churn probability", f"{avg_probability * 100:.1f}%")

    st.dataframe(scored, use_container_width=True, hide_index=True)
    st.download_button(
        "Download scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="telco_churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_model_details(artifacts):
    st.subheader("Model Details")

    col1, col2 = st.columns(2)
    col1.write("**Model artifact**")
    col1.code(artifacts.model_path.name)
    col1.write("**Model class**")
    col1.code(artifacts.model.__class__.__name__)

    col2.write("**Preprocessor artifact**")
    col2.code(artifacts.preprocessor_path.name)
    col2.write("**Preprocessor class**")
    col2.code(artifacts.preprocessor.__class__.__name__)

    st.write("**Expected input columns**")
    schema = pd.DataFrame({"column": expected_model_columns(artifacts)})
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.info(
        "Model artifacts are loaded from the repository at startup. Keep artifact filenames unchanged for Streamlit Cloud deployment."
    )


def main():
    inject_styles()
    render_header()

    try:
        artifacts = get_artifacts()
    except ArtifactLoadError as error:
        st.error("Model artifacts could not be loaded.")
        st.caption(str(error))
        st.stop()

    page = render_sidebar(artifacts)
    render_overview()
    st.divider()

    if page == "Single prediction":
        render_single_prediction(artifacts)
    elif page == "Batch scoring":
        render_batch_scoring(artifacts)
    else:
        render_model_details(artifacts)


if __name__ == "__main__":
    main()
