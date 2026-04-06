# This module trains the ML model for Smart Parking AI.
# Steps:
# 1. Import preprocessing function from preprocess.py
# 2. Load processed dataset
# 3. Define feature columns:
#    - hour
#    - day_of_week
#    - weekend
#    - temperature_c
#    - total_slots
#    - occupancy_rate
#    - encoded location_zone columns
#    - encoded weather columns
# 4. Target variable:
#    predicted_free_slots_next_30min
# 5. Split dataset into train/test (80/20)
# 6. Train RandomForestRegressor
# 7. Evaluate using MAE, RMSE, R2
# 8. Save model to model/rf_model.pkl using joblib

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import preprocessing function
from preprocess import preprocess_data

def get_feature_columns(df):
	base_features = ['hour', 'day_of_week', 'is_weekend', 'temperature_c', 'total_slots', 'occupancy_rate']
	# Find all one-hot encoded columns for location_zone and weather
	location_zone_cols = [col for col in df.columns if col.startswith('location_zone_')]
	weather_cols = [col for col in df.columns if col.startswith('weather_')]
	return base_features + location_zone_cols + weather_cols

def main():
	# Load processed dataset
	df = preprocess_data()

	# Drop rows with missing target
	df = df.dropna(subset=['predicted_free_slots_next_30min'])

	# Define features and target
	feature_cols = get_feature_columns(df)
	X = df[feature_cols]
	y = df['predicted_free_slots_next_30min']

	# Split dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train RandomForestRegressor
	model = RandomForestRegressor(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)

	# Predict and evaluate
	y_pred = model.predict(X_test)
	mae = mean_absolute_error(y_test, y_pred)
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	r2 = r2_score(y_test, y_pred)

	print(f"MAE: {mae:.2f}")
	print(f"RMSE: {rmse:.2f}")
	print(f"R2 Score: {r2:.2f}")

	# Save model
	model_dir = '../model'
	os.makedirs(model_dir, exist_ok=True)
	model_path = os.path.join(model_dir, 'rf_model.pkl')
	joblib.dump(model, model_path)
	print(f"Model saved to {model_path}")

if __name__ == '__main__':
	main()
