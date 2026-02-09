import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Load and preprocess data
# -----------------------------
data = pd.read_csv("exchange_rate.csv")
if 'date' in data.columns:
    data.drop(columns=['date'], inplace=True)
data.fillna(method='ffill', inplace=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

def create_sequences(data, window=30):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

# -----------------------------
# Rolling-origin CV
# -----------------------------
def rolling_cv(X, y, units_list, lr_list):
    results = []
    fold = int(0.15 * len(X))

    for units in units_list:
        for lr in lr_list:
            losses = []
            for i in range(fold, len(X)-fold, fold):
                X_tr, y_tr = X[:i], y[:i]
                X_val, y_val = X[i:i+fold], y[i:i+fold]

                model = Sequential([
                    LSTM(units, input_shape=(X.shape[1], X.shape[2])),
                    Dense(y.shape[1])
                ])
                model.compile(optimizer=Adam(lr), loss="mse")
                model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)
                losses.append(model.evaluate(X_val, y_val, verbose=0))

            results.append((units, lr, np.mean(losses)))
    return sorted(results, key=lambda x: x[2])

best = rolling_cv(X, y, [32, 64], [0.001, 0.0005])[0]
units, lr, _ = best

# -----------------------------
# Train final model
# -----------------------------
model = Sequential([
    LSTM(units, input_shape=(X.shape[1], X.shape[2])),
    Dense(y.shape[1])
])
model.compile(optimizer=Adam(lr), loss="mse")
model.fit(X, y, epochs=25, batch_size=32, verbose=1)

# -----------------------------
# Evaluation
# -----------------------------
pred = model.predict(X[-200:])
true = y[-200:]

pred_inv = scaler.inverse_transform(pred)
true_inv = scaler.inverse_transform(true)

mae = mean_absolute_error(true_inv, pred_inv)
rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))

print("MAE:", mae)
print("RMSE:", rmse)

# -----------------------------
# Visualization
# -----------------------------
plt.plot(true_inv[:,0], label="Actual")
plt.plot(pred_inv[:,0], label="Predicted")
plt.title("Actual vs Predicted (Primary Currency)")
plt.legend()
plt.show()

# -----------------------------
# SHAP Explainability
# -----------------------------
background = X[np.random.choice(X.shape[0], 50, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_vals = explainer.shap_values(X[-1:])

importance = np.mean(np.abs(shap_vals[0]), axis=1)[0]
plt.bar(range(len(importance)), importance)
plt.title("SHAP Feature Importance")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
