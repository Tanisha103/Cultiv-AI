import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import dpctl  # To specify device management for oneAPI
from sklearnex import patch_sklearn, config_context

# Apply the Intel optimizations for scikit-learn
patch_sklearn()

# Load the dataset (assuming the file is named crop_yield.csv)
data = pd.read_csv("crop_yield.csv")

# Select relevant columns
data = data[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']]

# Handle missing values
data = data.dropna()

# Convert categorical columns (Crop, Season, State) into numerical format using LabelEncoder
label_encoder_crop = LabelEncoder()
data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])

label_encoder_season = LabelEncoder()
data['Season'] = label_encoder_season.fit_transform(data['Season'])

label_encoder_state = LabelEncoder()
data['State'] = label_encoder_state.fit_transform(data['State'])

# Features and target
X = data[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']  # Yield is treated as the label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using Intel optimizations on a specific device (e.g., CPU)
with config_context(target_offload="auto"):
    model = LinearRegression()
    model.fit(X_train, y_train)

# Save the trained model and the encoders
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(label_encoder_crop, "label_encoder_crop.pkl")
joblib.dump(label_encoder_season, "label_encoder_season.pkl")
joblib.dump(label_encoder_state, "label_encoder_state.pkl")

# Optionally, you can evaluate the model's performance
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score}")
