
# This module handles data preprocessing for Smart Parking AI system.
# Steps:
# 1. Load dataset from data/parking_dataset.csv
# 2. Convert timestamp column to datetime
# 3. Extract:
#    - hour
#    - day_of_week
#    - weekend flag
# 4. One-hot encode:
#    - location_zone
#    - weather
# 5. Return processed dataframe ready for ML training
# Keep functions modular.

import pandas as pd

def load_dataset(path='C:/Users/Admin/OneDrive/Desktop/smart_parking_ai/data/smart_parking_ai_dataset.csv'):
	"""Load dataset from CSV file."""
	return pd.read_csv(path)

def convert_timestamp(df):
	"""Convert timestamp column to datetime format."""
	if 'timestamp' in df.columns:
		df['timestamp'] = pd.to_datetime(df['timestamp'])
	return df

def extract_time_features(df):
	"""Extract hour, day_of_week, and weekend flag from timestamp."""
	if 'timestamp' in df.columns:
		df['hour'] = df['timestamp'].dt.hour
		df['day_of_week'] = df['timestamp'].dt.dayofweek
		df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
	return df

def one_hot_encode(df, columns):
	"""One-hot encode specified columns."""
	return pd.get_dummies(df, columns=columns)

def preprocess_data(path='C:/Users/Admin/OneDrive/Desktop/smart_parking_ai/data/smart_parking_ai_dataset.csv'):
	"""Full preprocessing pipeline. Returns processed dataframe."""
	df = load_dataset(path)
	df = convert_timestamp(df)
	df = extract_time_features(df)
	# One-hot encode location_zone and weather if present
	encode_cols = [col for col in ['location_zone', 'weather'] if col in df.columns]
	df = one_hot_encode(df, encode_cols)
	return df

# Example usage
if __name__ == '__main__':
	processed_df = preprocess_data()
	print(processed_df.head())
