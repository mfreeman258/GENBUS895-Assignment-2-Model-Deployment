from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# Resolve artifact paths relative to this app.py file.
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "churn_rf_healthy_meals.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open(BASE_DIR / "churn_encoder_healthy_meals.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    return model, encoder


model, encoder = load_artifacts()


st.title("Customer Renewal Probability Predictor")
st.write(
    "Enter customer attributes and activity measures to predict the "
    "likelihood of subscription renewal."
)

st.subheader("Customer attributes")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1,
)

income_level = st.radio(
    "Income Level",
    ["Low", "Medium", "High", "Very High"],
)

education = st.radio(
    "Education",
    ["Graduate", "High School", "Other", "Post-Graduate"],
)

device_type = st.radio(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"],
)

tech_comfort_score = st.number_input(
    "Tech Comfort Score",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)

st.subheader("Customer activity")

total_num_sessions = st.number_input(
    "Total Number of Sessions",
    min_value=0,
    value=20,
    step=1,
)

gross_total_session_length = st.number_input(
    "Gross Total Session Length",
    min_value=0.0,
    value=600.0,
    step=1.0,
)

active_days = st.number_input(
    "Active Days",
    min_value=0,
    value=10,
    step=1,
)

active_quarters = st.number_input(
    "Active Quarters",
    min_value=0,
    value=2,
    step=1,
)

# This feature appears to have been engineered during model training.
avg_sessions_per_active_quarter = (
    total_num_sessions / active_quarters
    if active_quarters > 0
    else 0.0
)

st.caption(
    "Average sessions per active quarter: "
    f"{avg_sessions_per_active_quarter:.2f}"
)


if st.button("Predict", type="primary"):

    # Raw categorical fields must match the columns used to fit the encoder.
    raw_categorical = pd.DataFrame(
        [
            {
                "INCOME_LEVEL": income_level,
                "EDUCATION": education,
                "DEVICE_TYPE": device_type,
            }
        ]
    )

    encoded = encoder.transform(raw_categorical)
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(),
    )

    # Include every numeric feature used to train the random forest.
    numeric_df = pd.DataFrame(
        [
            {
                "TOTAL_NUM_SESSIONS": total_num_sessions,
                "GROSS_TOTAL_SESSION_LENGTH": gross_total_session_length,
                "ACTIVE_DAYS": active_days,
                "ACTIVE_QUARTERS": active_quarters,
                "AVG_SESSIONS_PER_ACTIVE_QUARTER": (
                    avg_sessions_per_active_quarter
                ),
                "AGE": age,
                "TECH_COMFORT_SCORE": tech_comfort_score,
            }
        ]
    )

    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    expected_features = list(model.feature_names_in_)
    provided_features = input_df.columns.tolist()

    missing_features = [
        feature
        for feature in expected_features
        if feature not in provided_features
    ]
    unexpected_features = [
        feature
        for feature in provided_features
        if feature not in expected_features
    ]

    if missing_features or unexpected_features:
        st.error(
            "The prediction data does not match the model's training features."
        )
        st.write("Missing features:", missing_features)
        st.write("Unexpected features:", unexpected_features)
        st.stop()

    # Scikit-learn requires the same feature order used during model fitting.
    input_df = input_df.loc[:, expected_features]

    class_one_positions = np.where(model.classes_ == 1)[0]
    if len(class_one_positions) != 1:
        st.error(
            "The model does not contain the expected binary class label 1."
        )
        st.stop()

    class_one_index = int(class_one_positions[0])
    renewal_probability = model.predict_proba(input_df)[0][class_one_index]

    risk = (
        "Low"
        if renewal_probability >= 0.60
        else "Medium"
        if renewal_probability >= 0.40
        else "High"
    )

    st.metric(
        "Renewal Probability",
        f"{renewal_probability:.1%}",
    )

    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
