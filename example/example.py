import numpy as np

from deep_tcn_tensorflow.model import DeepTCN
from deep_tcn_tensorflow.plots import plot

# Generate some time series
N = 500
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(3), cov=np.eye(3), size=N)
a = 10 + 10 * t + 10 * np.cos(2 * np.pi * (10 * t - 0.5)) + 1 * e[:, 0]
b = 20 + 20 * t + 20 * np.cos(2 * np.pi * (20 * t - 0.5)) + 2 * e[:, 1]
c = 30 + 30 * t + 30 * np.cos(2 * np.pi * (30 * t - 0.5)) + 3 * e[:, 2]
y = np.hstack([a.reshape(-1, 1), b.reshape(-1, 1), c.reshape(-1, 1)])

# Fit the model
model = DeepTCN(
    y=y,
    x=None,
    forecast_period=100,
    lookback_period=100,
    quantiles=[0.001, 0.1, 0.5, 0.9, 0.999],
    filters=3,
    kernel_size=3,
    dilation_rates=[1],
    loss='parametric'
)

model.fit(
    learning_rate=0.001,
    batch_size=16,
    epochs=300,
    verbose=1
)

# Generate the forecasts
df = model.forecast(y=y, x=None)

# Plot the forecasts
fig = plot(df=df, quantiles=[0.001, 0.1, 0.5, 0.9, 0.999])
fig.write_image('results.png', scale=4, height=900, width=700)