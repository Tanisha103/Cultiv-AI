import React, { useState } from "react";
import axios from "axios";
import './Predict.css';
function Predict() {
  const [inputData, setInputData] = useState({
    crop: "",
    crop_year: "",
    season: "",
    state: "",
    area: "",
    production: "",
    annual_rainfall: "",
    fertilizer: "",
    pesticide: ""
  });
  const [prediction, setPrediction] = useState(null);
  const handleChange = (e) => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };
  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://localhost:5000/predict", inputData);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error making prediction:", error);
    }
  };
  return (
    <div className="predict">
      <h1>Crop Yield Prediction</h1>
      <div className="input-container">
        <input
          name="crop"
          placeholder="Crop"
          value={inputData.crop}
          onChange={handleChange}
        />
        <input
          name="crop_year"
          placeholder="Crop Year"
          value={inputData.crop_year}
          onChange={handleChange}
        />
        <input
          name="season"
          placeholder="Season"
          value={inputData.season}
          onChange={handleChange}
        />
        <input
          name="state"
          placeholder="State"
          value={inputData.state}
          onChange={handleChange}
        />
        <input
          name="area"
          placeholder="Area (in hectares)"
          value={inputData.area}
          onChange={handleChange}
        />
        <input
          name="production"
          placeholder="Production"
          value={inputData.production}
          onChange={handleChange}
        />
        <input
          name="annual_rainfall"
          placeholder="Annual Rainfall (mm)"
          value={inputData.annual_rainfall}
          onChange={handleChange}
        />
        <input
          name="fertilizer"
          placeholder="Fertilizer (kg)"
          value={inputData.fertilizer}
          onChange={handleChange}
        />
        <input
          name="pesticide"
          placeholder="Pesticide (kg)"
          value={inputData.pesticide}
          onChange={handleChange}
        />
        <button onClick={handleSubmit}>Predict Yield</button>
      </div>
      {prediction && <h2>Predicted Yield: {prediction}</h2>}
    </div>
  );
}
export default Predict;
