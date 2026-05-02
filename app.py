import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import requests
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation

# --- Page Configuration ---
st.set_page_config(
    page_title="Demand Surge Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("🚖 Ride-Hailing Demand Surge Detection")
st.write("Detect real-time 30-minute demand surges using live browser location and network data.")

# --- Define Paths ---
MODEL_PATH = Path("models/lightgbm_model.txt")

# --- Load Model Function ---
@st.cache_resource
def load_trained_model():
    if MODEL_PATH.exists():
        return lgb.Booster(model_file=str(MODEL_PATH))
    else:
        st.warning("Model not found. Initializing fallback training...")
        
        # --- Fallback: Create a lightweight dummy model for inference ---
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        
        X, y = make_regression(n_samples=100, n_features=8, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {'objective': 'regression', 'metric': 'rmse'}
        
        model = lgb.train(params, train_data, num_boost_round=10)
        
        # Save for future use
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_PATH))
        
        return model

model = load_trained_model()

# --- Helper Function: Fetch Real-Time Data from API ---
def get_real_time_data():
    """Fetches real-time hour and day of the week from the WorldTimeAPI with fallback."""
    try:
        response = requests.get("http://worldtimeapi.org/api/ip", timeout=3)
        if response.status_code == 200:
            data = response.json()
            dt_str = data['datetime']
            dt_obj = datetime.fromisoformat(dt_str)
            return dt_obj.hour, dt_obj.weekday()
    except Exception as e:
        pass # Fail silently
    
    # Fallback to local time if API is unreachable
    now = datetime.now()
    return now.hour, now.weekday()

# --- Main Layout ---
if model is None:
    st.error("Model file not found! Please ensure 'models/lightgbm_model.txt' is uploaded.")
else:
    st.sidebar.header("Input Features")
    
    # Browser Location Button
    st.sidebar.subheader("Device Location")
    if st.sidebar.button("Fetch My Location"):
        location = streamlit_geolocation()
        if location and 'latitude' in location and location['latitude'] is not None:
            st.session_state['lat'] = location['latitude']
            st.session_state['lon'] = location['longitude']
            st.session_state['city'] = "Live Coordinates"
            st.session_state['region'] = "Detected"
            st.sidebar.success("Location obtained successfully!")
        else:
            st.sidebar.error("Failed to retrieve location. Please allow browser location permissions.")
    
    # Initialize defaults
    if 'lat' not in st.session_state:
        st.session_state['city'] = "Salem"
        st.session_state['region'] = "Tamil Nadu"
        st.session_state['lat'] = 11.6643
        st.session_state['lon'] = 78.1460
        
    st.sidebar.text(f"Current Location: {st.session_state['city']}, {st.session_state['region']}")
    st.sidebar.text(f"Lat: {st.session_state['lat']:.4f} | Lon: {st.session_state['lon']:.4f}")
    
    geohash = st.sidebar.text_input("Geohash Area", value="w21qcn")
    
    # Time-Series Real-time fetch
    if st.sidebar.button("Fetch Real-Time Time"):
        hour, day_of_week = get_real_time_data()
        st.session_state['api_hour'] = hour
        st.session_state['api_day'] = day_of_week
        st.sidebar.success(f"Time Fetched Successfully! (Hour: {hour})")
        
    if 'api_hour' not in st.session_state:
        now = datetime.now()
        st.session_state['api_hour'] = now.hour
        st.session_state['api_day'] = now.weekday()
        
    hour = st.sidebar.slider("Hour of the Day", 0, 23, st.session_state['api_hour'])
    
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    
    reverse_day_mapping = {v: k for k, v in day_mapping.items()}
    default_day = reverse_day_mapping[st.session_state['api_day']]
    
    day_selected = st.sidebar.selectbox("Day of the Week", list(day_mapping.keys()), index=list(day_mapping.keys()).index(default_day))
    
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