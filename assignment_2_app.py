from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# Locate the model files in the same folder as this script.
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    # Using the filenames visible in your file explorer
    with open(BASE_DIR / "churn_rf_assignment_2.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open(BASE_DIR / "churn_encoder_assignment_2.pkl", "rb") as encoder_file:
        encoder = pickle.load(encoder_file)

    return model, encoder

try:
    model, encoder = load_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes and activity measures to predict the likelihood of subscription renewal.")

st.subheader("Customer attributes")
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
income_level = st.radio("Income Level", ["Low", "Medium", "High", "Very High"])
education = st.radio("Education", ["Graduate", "High School", "Other", "Post-Graduate"])
device_type = st.radio("Device Type", ["Desktop-only", "Mobile-only", "Multi-device"])
tech_comfort_score = st.number_input("Tech Comfort Score", min_value=1, max_value=10, value=5, step=1)

st.subheader("Customer activity")
total_num_sessions = st.number_input("Total Number of Sessions", min_value=0, value=20, step=1)
gross_total_session_length = st.number_input("Gross Total Session Length", min_value=0.0, value=600.0, step=1.0)
active_days = st.number_input("Active Days", min_value=0, value=10, step=1)
active_quarters = st.number_input("Active Quarters", min_value=0, value=2, step=1)

avg_sessions_per_active_quarter = total_num_sessions / active_quarters if active_quarters > 0 else 0.0
st.caption(f"Average sessions per active quarter: {avg_sessions_per_active_quarter:.2f}")

if st.button("Predict", type="primary"):
    raw_categorical = pd.DataFrame([{"INCOME_LEVEL": income_level, "EDUCATION": education, "DEVICE_TYPE": device_type}])
    encoded = encoder.transform(raw_categorical)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    
    numeric_df = pd.DataFrame([{
        "TOTAL_NUM_SESSIONS": total_num_sessions,
        "GROSS_TOTAL_SESSION_LENGTH": gross_total_session_length,
        "ACTIVE_DAYS": active_days,
        "ACTIVE_QUARTERS": active_quarters,
        "AVG_SESSIONS_PER_ACTIVE_QUARTER": avg_sessions_per_active_quarter,
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_comfort_score
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    expected_features = list(model.feature_names_in_)
    input_df = input_df.loc[:, expected_features]

    renewal_probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if renewal_probability >= 0.60 else "Medium" if renewal_probability >= 0.40 else "High"

    st.metric("Renewal Probability", f"{renewal_probability:.1%}")
    if risk == "High": st.error(f"Churn Risk: {risk}")
    elif risk == "Medium": st.warning(f"Churn Risk: {risk}")
    else: st.success(f"Churn Risk: {risk}")