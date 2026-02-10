# ==========================================================
# Multivariate Time Series Forecasting using LSTM
# Dataset: UCI Exchange Rate (auto-loaded)
# Includes: Metrics, Visualization, SHAP Explainability
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

import shap

# ----------------------------------------------------------
# 1. Load Dataset (AUTO DOWNLOAD – no file needed)
# ----------------------------------------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/exchange_rate/exchange_rate.txt"
data = pd.read_csv(url, header=None)

values = data.values
n_features = values.shape[1]

print("Dataset loaded")
print("Shape:", values.shape)

# ----------------------------------------------------------
# 2. Scale Data
# ----------------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)

# ----------------------------------------------------------
# 3. Create Time-Series Sequences
# ----------------------------------------------------------
def create_sequences(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])   # multi-output
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# ----------------------------------------------------------
# 4. Train–Test Split (Time-Based)
# ----------------------------------------------------------
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ----------------------------------------------------------
# 5. Build LSTM Model
# ----------------------------------------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(n_features))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

# ----------------------------------------------------------
# 6. Train Model
# ----------------------------------------------------------
model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    verbose=0
)

print("Model training completed")

# ----------------------------------------------------------
# 7. Evaluation
# ----------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\nEvaluation Metrics")
print("------------------")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

# ----------------------------------------------------------
# 8. Visualization
# ----------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(y_test[:100, 0], label="Actual")
plt.plot(y_pred[:100, 0], label="Predicted")
plt.title("Actual vs Predicted (Feature 1)")
plt.xlabel("Time")
plt.ylabel("Scaled Value")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 9. SHAP Explainability
# ----------------------------------------------------------
# Use small background set for speed
background = X_train[:50]
test_instance = X_test[:1]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap
