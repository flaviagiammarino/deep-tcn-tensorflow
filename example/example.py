import numpy as np
from deep_tcn_tensorflow.model import DeepTCN

# Generate two time series
N = 1000
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=[0, 0], cov=[[15, 10.5], [10.5, 15]], size=N)
s = np.array([50 + 25 * np.cos(2 * np.pi * (10 * t - 0.5)), 100 + 50 * np.sin(2 * np.pi * (20 * t - 0.5))]).T
y = s + e

# Fit the model
model = DeepTCN(
    y=y,
    x=None,
    forecast_period=100,
    lookback_period=200,
    quantiles=[0.1, 0.5, 0.9],
    filters=8,
    kernel_size=2,
    dilation_rates=[1, 2, 4, 8],
)

model.fit(
    learning_rate=0.001,
    batch_size=64,
    epochs=100,
)

# Plot the in-sample predictions
predictions = model.predict(index=900)
predictions_ = model.plot_predictions()
predictions_.write_image('predictions.png', width=750, height=650)

# Plot the out of sample forecasts
forecasts = model.forecast()
forecasts_ = model.plot_forecasts()
forecasts_.write_image('forecasts.png', width=750, height=650)
