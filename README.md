# time_series_forecasting_using_LSTM
Advanced time series forecasting project using LSTM deep learning model on multivariate exchange rate data, including preprocessing, evaluation metrics, visualization, and explainability.

1. Dataset Description

The Exchange Rate Time Series dataset from the UCI Machine Learning Repository was used for this project.

The dataset consists of daily exchange rates of multiple currencies over time, making it a multivariate time series dataset. Each column represents a different currency exchange rate. The data exhibits strong temporal dependency, trend, and non-stationary behavior, which makes it suitable for deep learning-based forecasting.

2. Data Preprocessing

Missing values were handled using forward-fill to preserve temporal continuity.

The data was scaled using Min-Max normalization to bring all features into a uniform range, which improves LSTM training stability

3. Model Architecture

A Long Short-Term Memory (LSTM) neural network was selected due to its ability to capture long-term dependencies in time series data.

The model consists of:

One LSTM layer with 64 units

A Dropout layer to reduce overfitting

A Dense output layer to predict the next time step

The model was trained using the Adam optimizer and Mean Squared Error loss function.

4. Hyperparameter Tuning Strategy

Hyperparameters such as the number of epochs, batch size, and learning rate were selected based on validation performance. A walk-forward style validation was implicitly followed by maintaining the temporal order of the data during splitting.

5. Evaluation Results

The trained model was evaluated using standard forecasting metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Mean Absolute Percentage Error (MAPE)


6. Explainability Analysis

To understand the model's behavior, a feature importance analysis was performed by examining the average magnitude of input values across time steps.

This analysis highlighted which currency exchange rates contributed most significantly to the model's predictions.

Explainability improves trust in the model and provides insight into how different features influence forecasting outcomes.

7. Conclusion

This project demonstrated the application of deep learning for multivariate time series forecasting using an LSTM model.

The model successfully learned temporal dependencies and produced accurate forecasts.

Future improvements may include experimenting with Transformer-based models, more advanced explainability techniques such as SHAP, and multi-step forecasting.
