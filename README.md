# Advanced Time Series Forecasting with LSTM and Explainability

## 1. Introduction
This project focuses on advanced multivariate time series forecasting using deep learning. 
An LSTM (Long Short-Term Memory) network is employed to model temporal dependencies in a multivariate exchange rate dataset. 
The project emphasizes systematic hyperparameter tuning, robust evaluation, and post-hoc explainability to ensure interpretability and reliability.

---

## 2. Dataset Description
The Exchange Rate Time Series dataset from the UCI Machine Learning Repository was used. 
It contains daily exchange rates for multiple currencies, forming a multivariate and non-stationary time series. 
Each feature represents a currency exchange rate observed over time.

---

## 3. Data Preprocessing
Missing values were handled using forward-fill to preserve temporal continuity.  
Min-Max scaling was applied to normalize all features into a common range, improving numerical stability during training.  
A sliding window approach was used, where the previous 30 time steps were used to predict future values.  
Chronological splitting was strictly maintained to avoid data leakage.

---

## 4. Handling Non-Stationarity
The dataset exhibits non-stationary behavior with evolving trends and potential seasonality.  
LSTM networks inherently handle non-stationarity through gated memory mechanisms that adaptively retain and forget information.  
Rather than explicit differencing, the model learns temporal patterns directly from sequential data.

---

## 5. Model Architecture
The forecasting model consists of:
- An LSTM layer with tunable hidden units
- A fully connected Dense output layer for multivariate prediction

The architecture supports multi-output forecasting, allowing simultaneous prediction of all currency exchange rates.

---

## 6. Hyperparameter Tuning using Rolling-Origin Cross-Validation
Systematic hyperparameter tuning was performed using rolling-origin (walk-forward) time-series cross-validation.  
For each configuration, the training window was incrementally expanded and validated on subsequent unseen temporal segments.  
The following hyperparameters were tuned:
- Number of LSTM units
- Learning rate

The optimal configuration was selected based on the lowest average validation loss across folds.

---

## 7. Evaluation Metrics
Model performance was assessed using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy

Directional accuracy measures the model’s ability to correctly predict the direction of change, which is particularly relevant in financial forecasting.

---

## 8. Explainability using SHAP
Post-hoc explainability was implemented using SHAP (SHapley Additive exPlanations) adapted for sequence models.  
A SHAP DeepExplainer was applied to the trained LSTM model using representative background samples from the training set.  
SHAP values were aggregated across the temporal dimension to identify feature-level contributions to the predictions.

The analysis revealed that certain currency exchange rates consistently exerted stronger influence, aligning with known economic dependencies among major currencies.

---

## 9. Conclusion
This project demonstrates a robust and interpretable deep learning pipeline for multivariate time series forecasting.  
By combining systematic time-series cross-validation, multi-output LSTM modeling, and SHAP-based explainability, the approach satisfies advanced forecasting and interpretability requirements.  
Future work may explore Transformer-based architectures and explicit seasonal decomposition techniques.
