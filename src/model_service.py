from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd

from src.config import (
    CONTRACT_OPTIONS,
    FEATURE_COLUMNS,
    HIGH_RISK_THRESHOLD,
    INTERNET_SERVICE_OPTIONS,
    LOW_RISK_THRESHOLD,
    MULTIPLE_LINES_OPTIONS,
    NUMERIC_COLUMNS,
    PAYMENT_METHOD_OPTIONS,
    RECOMMENDED_ACTIONS,
    SERVICE_OPTIONS,
    YES_NO_OPTIONS,
)


class ArtifactLoadError(RuntimeError):
    """Raised when trained artifacts cannot be loaded."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: object
    preprocessor: object
    model_path: Path
    preprocessor_path: Path


@dataclass(frozen=True)
class PredictionResult:
    prediction_label: str
    churn_probability: float | None
    risk_level: str
    recommended_action: str


def load_artifacts(model_path: Path, preprocessor_path: Path) -> ModelArtifacts:
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(preprocessor_path, "rb") as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
    except FileNotFoundError as error:
        raise ArtifactLoadError(f"Missing artifact: {error.filename}") from error
    except Exception as error:
        raise ArtifactLoadError(str(error)) from error

    return ModelArtifacts(
        model=model,
        preprocessor=preprocessor,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )


def expected_model_columns(artifacts: ModelArtifacts) -> list[str]:
    if hasattr(artifacts.preprocessor, "feature_names_in_"):
        return [str(column) for column in artifacts.preprocessor.feature_names_in_]
    return FEATURE_COLUMNS


def validate_customer(customer: dict) -> list[str]:
    errors = []

    missing = [column for column in FEATURE_COLUMNS if column not in customer]
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")

    if customer.get("gender") not in ["Female", "Male"]:
        errors.append("Gender must be Female or Male.")
    if customer.get("SeniorCitizen") not in [0, 1]:
        errors.append("SeniorCitizen must be 0 or 1.")
    if customer.get("Partner") not in YES_NO_OPTIONS:
        errors.append("Partner must be No or Yes.")
    if customer.get("Dependents") not in YES_NO_OPTIONS:
        errors.append("Dependents must be No or Yes.")
    if customer.get("PhoneService") not in YES_NO_OPTIONS:
        errors.append("PhoneService must be No or Yes.")
    if customer.get("MultipleLines") not in MULTIPLE_LINES_OPTIONS:
        errors.append("MultipleLines has an unsupported value.")
    if customer.get("InternetService") not in INTERNET_SERVICE_OPTIONS:
        errors.append("InternetService has an unsupported value.")
    if customer.get("Contract") not in CONTRACT_OPTIONS:
        errors.append("Contract has an unsupported value.")
    if customer.get("PaymentMethod") not in PAYMENT_METHOD_OPTIONS:
        errors.append("PaymentMethod has an unsupported value.")

    for column in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        if customer.get(column) not in SERVICE_OPTIONS:
            errors.append(f"{column} has an unsupported value.")

    for column in NUMERIC_COLUMNS:
        try:
            value = float(customer.get(column))
        except (TypeError, ValueError):
            errors.append(f"{column} must be numeric.")
            continue

        if value < 0:
            errors.append(f"{column} cannot be negative.")

    if float(customer.get("tenure", 0)) > 100:
        errors.append("Tenure is above the supported range for this app.")
    if float(customer.get("MonthlyCharges", 0)) > 1000:
        errors.append("MonthlyCharges is above the supported range for this app.")
    if float(customer.get("TotalCharges", 0)) > 100000:
        errors.append("TotalCharges is above the supported range for this app.")

    return errors


def build_customer_frame(customer: dict) -> pd.DataFrame:
    normalized = {column: customer[column] for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([normalized], columns=FEATURE_COLUMNS)
    return coerce_frame_types(frame)


def coerce_frame_types(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    return frame


def positive_class_index(model: object) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1

    normalized_classes = [str(value).strip().lower() for value in classes]
    for candidate in ["1", "yes", "true", "churn"]:
        if candidate in normalized_classes:
            return normalized_classes.index(candidate)

    return min(1, len(normalized_classes) - 1)


def prediction_to_label(prediction: object) -> str:
    normalized = str(prediction).strip().lower()
    return "Yes" if normalized in {"1", "yes", "true", "churn"} else "No"


def probability_to_risk(probability: float | None, prediction_label: str) -> str:
    if probability is None or pd.isna(probability):
        return "High" if prediction_label == "Yes" else "Low"
    if probability < LOW_RISK_THRESHOLD:
        return "Low"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium"
    return "High"


def score_customer(artifacts: ModelArtifacts, customer_frame: pd.DataFrame) -> PredictionResult:
    customer_frame = coerce_frame_types(customer_frame[FEATURE_COLUMNS])
    transformed = artifacts.preprocessor.transform(customer_frame)
    prediction = artifacts.model.predict(transformed)[0]
    prediction_label = prediction_to_label(prediction)

    churn_probability = None
    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(transformed)
        churn_probability = float(probabilities[0][positive_class_index(artifacts.model)])

    risk_level = probability_to_risk(churn_probability, prediction_label)
    return PredictionResult(
        prediction_label=prediction_label,
        churn_probability=churn_probability,
        risk_level=risk_level,
        recommended_action=RECOMMENDED_ACTIONS[risk_level],
    )


def score_batch(artifacts: ModelArtifacts, batch: pd.DataFrame) -> pd.DataFrame:
    batch = coerce_frame_types(batch[FEATURE_COLUMNS])
    transformed = artifacts.preprocessor.transform(batch)
    predictions = artifacts.model.predict(transformed)

    scored = batch.copy()
    scored["prediction"] = [prediction_to_label(prediction) for prediction in predictions]

    if hasattr(artifacts.model, "predict_proba"):
        index = positive_class_index(artifacts.model)
        probabilities = artifacts.model.predict_proba(transformed)[:, index]
        scored["churn_probability"] = probabilities
    else:
        scored["churn_probability"] = pd.NA

    scored["risk_level"] = [
        probability_to_risk(probability, label)
        for probability, label in zip(scored["churn_probability"], scored["prediction"])
    ]

    return scored
