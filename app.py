import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearnex import patch_sklearn, config_context
patch_sklearn()
app = Flask(__name__)
model = joblib.load("crop_yield_model.pkl")
label_encoder_crop = joblib.load("label_encoder_crop.pkl")
label_encoder_season = joblib.load("label_encoder_season.pkl")
label_encoder_state = joblib.load("label_encoder_state.pkl")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    crop = data["crop"]
    crop_year = float(data["crop_year"])
    season = data["season"]
    state = data["state"]
    area = float(data["area"])
    production = float(data["production"])
    annual_rainfall = float(data["annual_rainfall"])
    fertilizer = float(data["fertilizer"])
    pesticide = float(data["pesticide"])
    crop_encoded = label_encoder_crop.transform([crop])[0]
    season_encoded = label_encoder_season.transform([season])[0]
    state_encoded = label_encoder_state.transform([state])[0]
    features = np.array([[crop_encoded, crop_year, season_encoded, state_encoded, area, production, annual_rainfall, fertilizer, pesticide]])
    with config_context(target_offload="auto"):
        prediction = model.predict(features)[0]
    return jsonify({"prediction": prediction})
if __name__ == "__main__":
    app.run(debug=True)
