import numpy as np
from deep_tcn_tensorflow.model import DeepTCN

# Generate two time series
N = 1000
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.25], [0.25, 1]], size=N)
a = 40 + 30 * t + 20 * np.cos(2 * np.pi * (10 * t - 0.5)) + e[:, 0]
b = 50 + 40 * t + 30 * np.sin(2 * np.pi * (20 * t - 0.5)) + e[:, 1]
y = np.hstack([a.reshape(- 1, 1), b.reshape(- 1, 1)])

# Fit the model
model = DeepTCN(
    y=y,
    x=None,
    forecast_period=100,
    lookback_period=200,
    quantiles=[0.005, 0.05, 0.5, 0.95, 0.995],
    filters=4,
    kernel_size=3,
    dilation_rates=[1, 2],
    loss='nonparametric'
)

model.fit(
    learning_rate=0.01,
    batch_size=64,
    epochs=200,
)

# Plot the in-sample predictions
predictions = model.predict(index=900)
predictions_ = model.plot_predictions()
predictions_.write_image('predictions.png', width=750, height=650)

# Plot the out of sample forecasts
forecasts = model.forecast()
forecasts_ = model.plot_forecasts()
forecasts_.write_image('forecasts.png', width=750, height=650)