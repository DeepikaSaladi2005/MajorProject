# train_cost_model_json.py

import json
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ===========================
# 1. Load JSON Dataset
# ===========================
with open("car_parts_prices.json") as f:
    data = json.load(f)

rows = []
for brand, models in data.items():
    for model_name, parts in models.items():
        for part, cost in parts.items():
            rows.append({
                "car": brand,
                "model": model_name,
                "body part": part,
                "avg_cost": cost
            })

df = pd.DataFrame(rows)
print(f"✅ Loaded JSON. Dataset shape: {df.shape}")

# ===========================
# 2. Features & Target
# ===========================
X = df[["car", "model", "body part"]]
y = df["avg_cost"]

# ===========================
# 3. Preprocessing
# ===========================
# One-hot encode all categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["car", "model", "body part"])
    ]
)

# ===========================
# 4. Build ML Pipeline
# ===========================
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# ===========================
# 5. Train-Test Split & Train
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ===========================
# 6. Evaluate
# ===========================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Training complete! Mean Absolute Error: {mae:.2f}")

# ===========================
# 7. Save Trained Model
# ===========================
os.makedirs("weights", exist_ok=True)
joblib.dump(model, "weights/repair_cost_model.pkl")
print("💾 Model saved to weights/repair_cost_model.pkl")
