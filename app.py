import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand Surge Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🚖 Ride-Hailing Demand Surge Detection")
st.write("Detect real-time 30-minute demand surges using live location and network data.")

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

# --- Helper Function: Fetch Real Location via IP ---
def get_real_location():
    """Fetches real-time city and coordinates using the ip-api service."""
    try:
        # Get location based on IP address
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return data.get('city'), data.get('regionName'), data.get('lat'), data.get('lon')
    except Exception as e:
        st.sidebar.warning("Failed to connect to location API. Using default values.")
    
    # Fallback to default coordinates (e.g., Chennai, Tamil Nadu)
    return "Chennai", "Tamil Nadu", 13.0827, 80.2707

# --- Main Layout ---
if model is None:
    st.error("Model file not found! Please ensure 'models/lightgbm_model.txt' is uploaded.")
else:
    st.sidebar.header("Input Features")
    
    # Trigger location API
    if st.sidebar.button("Fetch My Real Location"):
        city, region, lat, lon = get_real_location()
        st.session_state['city'] = city
        st.session_state['region'] = region
        st.session_state['lat'] = lat
        st.session_state['lon'] = lon
        st.sidebar.success(f"Location Found: {city}, {region}")
    
    # Initialize session defaults
    if 'lat' not in st.session_state:
        st.session_state['city'] = "Chennai"
        st.session_state['region'] = "Tamil Nadu"
        st.session_state['lat'] = 13.0827
        st.session_state['lon'] = 80.2707
        
    st.sidebar.text(f"Current Location: {st.session_state['city']}, {st.session_state['region']}")
    st.sidebar.text(f"Lat: {st.session_state['lat']:.4f} | Lon: {st.session_state['lon']:.4f}")
    
    geohash = st.sidebar.text_input("Geohash Area", value="w21qcn")
    
    hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
    
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    
    day_selected = st.sidebar.selectbox("Day of the Week", list(day_mapping.keys()))
    
    st.sidebar.subheader("Time-Series Features")
    lag_1 = st.sidebar.number_input("Lag 1 (Demand 1 step ago)", value=0.10, step=0.01)
    lag_2 = st.sidebar.number_input("Lag 2 (Demand 2 steps ago)", value=0.10, step=0.01)
    lag_3 = st.sidebar.number_input("Lag 3 (Demand 3 steps ago)", value=0.10, step=0.01)
    rolling_mean_3 = st.sidebar.number_input("Rolling Mean (3 steps)", value=0.10, step=0.01)

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
        prediction = model.predict(input_df)
        demand_pred = prediction[0]

        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Predicted Demand Level", value=f"{demand_pred:.4f}")
            
        with col2:
            if demand_pred > 0.25:
                st.error("🚨 **High Demand Surge Expected!**")
            else:
                st.success("🟢 **Normal Demand Expected.**")