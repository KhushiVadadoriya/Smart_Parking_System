# Streamlit App for Smart Parking AI System
# App should:
# 1. Load trained model
# 2. Provide user inputs: location_zone, hour, weather, temperature
# 3. When user clicks Predict: preprocess input, predict, classify congestion, recommend alternative
# 4. Display: predicted free slots, congestion level (color-coded), recommendation
# Keep UI simple and clean.


import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from src.recommend import load_model, predict_next_30min, classify_congestion, recommend_zone

# Use absolute path for model and feature list
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_model_features.pkl')

st.title("Smart Parking AI System")
st.write("Predict parking availability and get recommendations.")


# Load feature list if available, else fallback to defaults
default_location_zones = ['Zone_A', 'Zone_B', 'Zone_C']
default_weather_options = ['Clear', 'Rainy', 'Cloudy', 'Snowy']
feature_list = None
location_zones = default_location_zones
weather_options = default_weather_options
if os.path.exists(FEATURES_PATH):
	try:
		feature_list = joblib.load(FEATURES_PATH)
		# Extract zones and weather from feature names
		location_zones = sorted({f.split('location_zone_')[1] for f in feature_list if f.startswith('location_zone_')}) or default_location_zones
		weather_options = sorted({f.split('weather_')[1] for f in feature_list if f.startswith('weather_')}) or default_weather_options
	except Exception:
		feature_list = None

# Try loading the model, show error if not found
model = None
model_load_error = None
if os.path.exists(MODEL_PATH):
	try:
		model = load_model(MODEL_PATH)
	except Exception as e:
		model_load_error = str(e)
else:
	model_load_error = f"Model file not found at {MODEL_PATH}. Please train the model first."

# User inputs
zone = st.selectbox("Select Location Zone", location_zones)
hour = st.slider("Hour of Day", 0, 23, 12)
weather = st.selectbox("Weather", weather_options)
temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0)

if model_load_error:
	st.error(model_load_error)
	st.button("Predict", disabled=True)
else:
	if st.button("Predict"):
		try:
			# Example: Assume total_slots and occupancy_rate are fixed for demo
			total_slots = 100
			occupancy_rate = 0.5

			# One-hot encode zone and weather
			input_dict = {
				'hour': hour,
				'day_of_week': pd.Timestamp.now().dayofweek,
				'is_weekend': int(pd.Timestamp.now().dayofweek in [5, 6]),
				'temperature_c': temperature,
				'total_slots': total_slots,
				'occupancy_rate': occupancy_rate,
			}
			for z in location_zones:
				input_dict[f'location_zone_{z}'] = int(z == zone)
			for w in weather_options:
				input_dict[f'weather_{w}'] = int(w == weather)

			# Use feature_list if available, else fallback to sorted keys
			if feature_list:
				input_features = [input_dict.get(k, 0) for k in feature_list]
			else:
				input_features = [input_dict[k] for k in sorted(input_dict.keys())]

			# Predict free slots
			predicted_free = int(predict_next_30min(model, input_features))
			congestion = classify_congestion(predicted_free, total_slots)

			# Simulate predictions for all zones for recommendation
			zone_predictions = {}
			for z in location_zones:
				test_dict = input_dict.copy()
				for zz in location_zones:
					test_dict[f'location_zone_{zz}'] = int(zz == z)
				if feature_list:
					test_features = [test_dict.get(k, 0) for k in feature_list]
				else:
					test_features = [test_dict[k] for k in sorted(test_dict.keys())]
				zone_predictions[z] = (int(predict_next_30min(model, test_features)), total_slots)

			rec_result = recommend_zone(zone_predictions, zone)

			# Color map for congestion
			color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Unknown': 'gray'}
			st.markdown(f"<h3>Predicted Free Slots: <span style='color:blue'>{predicted_free}</span></h3>", unsafe_allow_html=True)
			st.markdown(f"<h3>Congestion Level: <span style='color:{color_map.get(congestion, 'gray')}'>{congestion}</span></h3>", unsafe_allow_html=True)
			if rec_result['recommendation']:
				st.info(f"Congestion is high in {zone}. Recommended alternative: {rec_result['recommendation']}")
			else:
				st.success(f"{zone} is suitable for parking.")
		except Exception as e:
			st.error(f"Prediction failed: {e}")
            
st.write("Model loaded:", model)        
