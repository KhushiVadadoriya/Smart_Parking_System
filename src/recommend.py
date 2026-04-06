# This module handles prediction and recommendation logic.

import os
import joblib
import numpy as np


# ===============================
# 1️⃣ Load Model Safely
# ===============================
def load_model():
    """
    Load trained RandomForest model from model/rf_model.pkl
    Uses dynamic path resolution.
    """
    try:
        # Get project root directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "model", "rf_model.pkl")

        if not os.path.exists(model_path):
            print("Model file not found at:", model_path)
            return None

        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model

    except Exception as e:
        print("Error loading model:", e)
        return None


# ===============================
# 2️⃣ Prediction Function
# ===============================
def predict_next_30min(model, input_features):
    """
    Predict free slots for next 30 minutes.
    input_features must match training feature order.
    """
    if model is None:
        raise ValueError("Model is not loaded properly.")

    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)

    return float(prediction[0])


# ===============================
# 3️⃣ Congestion Classification
# ===============================
def classify_congestion(free_slots, total_slots):
    """
    Classify congestion level based on occupancy percentage.
    """
    if total_slots == 0:
        return "Unknown"

    occupancy_pct = 100 * (1 - free_slots / total_slots)

    if occupancy_pct < 50:
        return "Low"
    elif occupancy_pct <= 80:
        return "Medium"
    else:
        return "High"


# ===============================
# 4️⃣ Smart Zone Recommendation
# ===============================
def recommend_zone(zone_predictions, selected_zone):
    """
    zone_predictions format:
    {
        "Zone_A": (predicted_free_slots, total_slots),
        "Zone_B": (predicted_free_slots, total_slots)
    }
    """

    # Rank zones by highest free slots
    ranked = sorted(
        zone_predictions.items(),
        key=lambda x: x[1][0],
        reverse=True
    )

    selected_free, selected_total = zone_predictions.get(
        selected_zone, (0, 0)
    )

    congestion = classify_congestion(selected_free, selected_total)

    recommendation = None

    if congestion == "High":
        for zone, (free, total) in ranked:
            if zone != selected_zone:
                if classify_congestion(free, total) != "High":
                    recommendation = zone
                    break

    return {
        "ranked_zones": [zone for zone, _ in ranked],
        "selected_congestion": congestion,
        "recommendation": recommendation
    }
