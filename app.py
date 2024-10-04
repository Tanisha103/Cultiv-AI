import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearnex import patch_sklearn, config_context

# Apply Intel optimizations for scikit-learn
patch_sklearn()

app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load("crop_yield_model.pkl")
label_encoder_crop = joblib.load("label_encoder_crop.pkl")
label_encoder_season = joblib.load("label_encoder_season.pkl")
label_encoder_state = joblib.load("label_encoder_state.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.get_json()

    # Extract features from the request data
    crop = data["crop"]
    crop_year = float(data["crop_year"])
    season = data["season"]
    state = data["state"]
    area = float(data["area"])
    production = float(data["production"])
    annual_rainfall = float(data["annual_rainfall"])
    fertilizer = float(data["fertilizer"])
    pesticide = float(data["pesticide"])

    # Encode categorical data using the label encoders
    crop_encoded = label_encoder_crop.transform([crop])[0]
    season_encoded = label_encoder_season.transform([season])[0]
    state_encoded = label_encoder_state.transform([state])[0]

    # Prepare the feature vector for the model
    features = np.array([[crop_encoded, crop_year, season_encoded, state_encoded, area, production, annual_rainfall, fertilizer, pesticide]])

    # Make a prediction using Intel optimizations
    with config_context(target_offload="auto"):
        prediction = model.predict(features)[0]

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
