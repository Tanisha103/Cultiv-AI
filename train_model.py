import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import dpctl
from sklearnex import patch_sklearn, config_context
patch_sklearn()
data = pd.read_csv("crop_yield.csv")
data = data[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']]
data = data.dropna()
label_encoder_crop = LabelEncoder()
data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])
label_encoder_season = LabelEncoder()
data['Season'] = label_encoder_season.fit_transform(data['Season'])
label_encoder_state = LabelEncoder()
data['State'] = label_encoder_state.fit_transform(data['State'])
X = data[['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
y = data['Yield']  # Yield is treated as the label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
with config_context(target_offload="auto"):
    model = LinearRegression()
    model.fit(X_train, y_train)
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(label_encoder_crop, "label_encoder_crop.pkl")
joblib.dump(label_encoder_season, "label_encoder_season.pkl")
joblib.dump(label_encoder_state, "label_encoder_state.pkl")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")
