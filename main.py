# ==========================================
# Advanced Time Series Forecasting using LSTM
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Download "exchange_rate.csv" from:
# https://archive.ics.uci.edu/ml/datasets/Exchange+Rate+Dataset

data = pd.read_csv("exchange_rate.csv")

# Drop date column if exists
if 'date' in data.columns:
    data = data.drop(columns=['date'])

# Handle missing values
data.fillna(method='ffill', inplace=True)

print("Dataset shape:", data.shape)

# -------------------------------
# 2. Scale Data
# -------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------
# 3. Create Sequences
# -------------------------------
def create_sequences(data, window_size=30):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])  # predict first feature
    return np.array(X), np.array(y)

WINDOW_SIZE = 30
X, y = create_sequences(scaled_data, WINDOW_SIZE)

# -------------------------------
# 4. Train / Val / Test Split
# -------------------------------
train_size = int(0.7 * len(X))
val_size = int(0.85 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:val_size], y[train_size:val_size]
X_test, y_test = X[val_size:], y[val_size:]

# -------------------------------
# 5. Build LSTM Model
# -------------------------------
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

model.summary()

# -------------------------------
# 6. Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# Inverse scaling
y_test_inv = scaler.inverse_transform(
    np.hstack([y_test.reshape(-1,1),
               np.zeros((len(y_test), scaled_data.shape[1]-1))])
)[:,0]

y_pred_inv = scaler.inverse_transform(
    np.hstack([y_pred,
               np.zeros((len(y_pred), scaled_data.shape[1]-1))])
)[:,0]

# -------------------------------
# 8. Evaluation Metrics
# -------------------------------
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

direction_true = np.sign(np.diff(y_test_inv))
direction_pred = np.sign(np.diff(y_pred_inv))
directional_accuracy = np.mean(direction_true == direction_pred)

print("\nEvaluation Metrics")
print("------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("Directional Accuracy:", directional_accuracy)

# -------------------------------
# 9. Visualization
# -------------------------------

# Training vs Validation Loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# Actual vs Predicted
plt.figure()
plt.plot(y_test_inv[:200], label='Actual')
plt.plot(y_pred_inv[:200], label='Predicted')
plt.xlabel("Time Steps")
plt.ylabel("Exchange Rate")
plt.title("Actual vs Predicted Exchange Rate")
plt.legend()
plt.show()

# -------------------------------
# 10. Simple Explainability
# -------------------------------
feature_importance = np.mean(np.abs(X_train), axis=(0,1))

plt.figure()
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance (Approximation)")
plt.show()
