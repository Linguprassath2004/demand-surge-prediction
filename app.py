import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand Surge Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🚖 Ride-Hailing Demand Surge Prediction")
st.write("Predict 30-minute demand surges using your trained LightGBM model.")

# --- Define Paths ---
MODEL_PATH = Path("models/lightgbm_model.txt")

# --- Load Model Function ---
@st.cache_resource
def load_trained_model():
    if MODEL_PATH.exists():
        return lgb.Booster(model_file=str(MODEL_PATH))
    else:
        return None

model = load_trained_model()

# --- Main Layout ---
if model is None:
    st.error("Model file not found! Please train the model first using `main.py`.")
else:
    # Sidebar for Inputs
    st.sidebar.header("Input Features")
    
    geohash = st.sidebar.text_input("Geohash", value="w21qcn")
    hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
    
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    day_selected = st.sidebar.selectbox("Day of the Week", list(day_mapping.keys()))
    
    st.sidebar.subheader("Time-Series Features")
    lag_1 = st.sidebar.number_input("Lag 1 (Demand 1 step ago)", value=0.10)
    lag_2 = st.sidebar.number_input("Lag 2 (Demand 2 steps ago)", value=0.10)
    lag_3 = st.sidebar.number_input("Lag 3 (Demand 3 steps ago)", value=0.10)
    rolling_mean_3 = st.sidebar.number_input("Rolling Mean (3 steps)", value=0.10)

    # Prepare input dataframe
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    input_df = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_mapping[day_selected]],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'lag_1': [lag_1],
        'lag_2': [lag_2],
        'lag_3': [lag_3],
        'rolling_mean_3': [rolling_mean_3]
    })

    if st.sidebar.button("Predict Surge"):
        # Ensure feature order matches the training dataset
        prediction = model.predict(input_df)
        demand_pred = prediction[0]

        # Results dashboard
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Demand Level", value=f"{demand_pred:.4f}")
            
        with col2:
            if demand_pred > 0.25:
                st.error("🚨 **High Demand Surge Expected!**")
            else:
                st.success("🟢 **Normal Demand Expected.**")