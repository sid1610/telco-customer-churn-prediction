from pathlib import Path

import pandas as pd
import streamlit as st
import pickle


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "telco_churn_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(PREPROCESSOR_PATH, "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    return model, preprocessor


def yes_no(label, default="No"):
    return st.selectbox(label, ["No", "Yes"], index=["No", "Yes"].index(default))


def service_choice(label, default="No"):
    options = ["No", "No internet service", "Yes"]
    return st.selectbox(label, options, index=options.index(default))


def build_customer_dataframe(inputs):
    return pd.DataFrame(
        [
            {
                "gender": inputs["gender"],
                "SeniorCitizen": int(inputs["senior_citizen"]),
                "Partner": inputs["partner"],
                "Dependents": inputs["dependents"],
                "tenure": inputs["tenure"],
                "PhoneService": inputs["phone_service"],
                "MultipleLines": inputs["multiple_lines"],
                "InternetService": inputs["internet_service"],
                "OnlineSecurity": inputs["online_security"],
                "OnlineBackup": inputs["online_backup"],
                "DeviceProtection": inputs["device_protection"],
                "TechSupport": inputs["tech_support"],
                "StreamingTV": inputs["streaming_tv"],
                "StreamingMovies": inputs["streaming_movies"],
                "Contract": inputs["contract"],
                "PaperlessBilling": inputs["paperless_billing"],
                "PaymentMethod": inputs["payment_method"],
                "MonthlyCharges": float(inputs["monthly_charges"]),
                "TotalCharges": float(inputs["total_charges"]),
            }
        ]
    )


def churn_probability(model, transformed_data):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_data)
        if probabilities.shape[1] > 1:
            return float(probabilities[0][1])
        return float(probabilities[0][0])
    return None


st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="wide",
)

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to estimate whether the customer is likely to churn.")

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Missing model file: {error.filename}")
    st.stop()
except Exception as error:
    st.error("Could not load the model artifacts.")
    st.caption(str(error))
    st.stop()

with st.form("prediction_form"):
    left, middle, right = st.columns(3)

    with left:
        st.subheader("Customer")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.checkbox("Senior Citizen")
        partner = yes_no("Partner")
        dependents = yes_no("Dependents")
        tenure = st.number_input("Tenure in months", min_value=0, max_value=100, value=12)

    with middle:
        st.subheader("Services")
        phone_service = yes_no("Phone Service", "Yes")
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "No phone service", "Yes"],
            index=0,
        )
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = service_choice("Online Security")
        online_backup = service_choice("Online Backup")
        device_protection = service_choice("Device Protection")
        tech_support = service_choice("Tech Support")
        streaming_tv = service_choice("Streaming TV")
        streaming_movies = service_choice("Streaming Movies")

    with right:
        st.subheader("Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = yes_no("Paperless Billing", "Yes")
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            index=2,
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

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if submitted:
    input_data = build_customer_dataframe(
        {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
        }
    )

    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)[0]
    probability = churn_probability(model, transformed_data)

    churn_label = "Yes" if str(prediction).lower() in {"1", "yes", "true"} else "No"

    result_col, probability_col = st.columns(2)
    with result_col:
        if churn_label == "Yes":
            st.error("Prediction: Customer is likely to churn")
        else:
            st.success("Prediction: Customer is unlikely to churn")

    with probability_col:
        if probability is not None:
            st.metric("Estimated churn probability", f"{probability * 100:.1f}%")
        else:
            st.metric("Predicted class", churn_label)

    with st.expander("View model input"):
        st.dataframe(input_data, use_container_width=True)
