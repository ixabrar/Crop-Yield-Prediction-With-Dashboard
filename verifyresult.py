
import joblib
import pandas as pd

# Load artifacts
model = joblib.load("best_reg_model.pkl")
scaler = joblib.load("scaler_reg.pkl")
le_crop = joblib.load("le_item.pkl")
le_state = joblib.load("le_state.pkl")
le_season = joblib.load("le_season.pkl")

# Same inputs as UI
data = {
    "Crop": le_crop.transform(["Banana"])[0],
    "Crop_Year": 2027,
    "Season": le_season.transform(["Rabi"])[0],
    "State": le_state.transform(["Delhi"])[0],
    "Annual_Rainfall": 123.0,
    "Fertilizer": 0.25,
    "Pesticide": 0.22
}

df = pd.DataFrame([data])
df_scaled = scaler.transform(df)

pred = model.predict(df_scaled)
print("Predicted Yield:", pred[0])
