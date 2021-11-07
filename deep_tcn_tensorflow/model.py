import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
pd.options.mode.chained_assignment = None

from deep_tcn_tensorflow.modules import encoder, decoder
from deep_tcn_tensorflow.utils import get_training_sequences_with_covariates, get_training_sequences
from deep_tcn_tensorflow.losses import parametric_loss, nonparametric_loss
from deep_tcn_tensorflow.plots import plot

class DeepTCN():

    def __init__(self,
                 y,
                 x=None,
                 forecast_period=1,
                 lookback_period=2,
                 quantiles=[0.1, 0.5, 0.9],
                 filters=32,
                 kernel_size=2,
                 dilation_rates=[1, 2, 4, 8],
                 units=64,
                 loss='nonparametric'):

        '''
        Implementation of multivariate time series forecasting model introduced in Chen, Y., Kang, Y., Chen, Y., &
        Wang, Z. (2020). Probabilistic forecasting with temporal convolutional neural network. Neurocomputing, 399,
        491-501. https://doi.org/10.1016/j.neucom.2020.03.011.

        Parameters:
        __________________________________
        y: np.array
            Target time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
            and n_targets is the number of target time series.

        x: np.array
            Features time series, array with shape (n_samples, n_features) where n_samples is the length of the time series
            and n_features is the number of features time series.

        forecast_period: int
            Length of forecast period (or decoder length).

        lookback_period: int
            Length of lookback period (or encoder length).

        quantiles: list.
            Quantiles of target time series to be predicted.

        filters: int.
            Number of filters (or channels) of the convolutional layers in the encoder module.

        kernel_size: int.
            Kernel size of the convolutional layers in the encoder module.

        dilation_rates: list.
            Dilation rates of the convolutional layers in the encoder module.

        units: int.
            Hidden units of dense layers in the decoder module.

        loss: str.
            The loss function, either 'nonparametric' or 'parametric'.
        '''

        if type(y) != np.ndarray:
            raise ValueError('The target time series must be provided as a numpy array.')

        if len(y.shape) > 2:
            raise ValueError('The targets array cannot have more than 2 dimensions. Found {} dimensions.'.format(len(y.shape)))

        elif len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        if x is not None:

            if type(x) != np.ndarray:
                raise ValueError('The features time series must be provided as a numpy array.')

            if len(x.shape) > 2:
                raise ValueError('The features array cannot have more than 2 dimensions. Found {} dimensions.'.format(len(x.shape)))

            elif len(x.shape) == 1:
                x = np.expand_dims(x, axis=1)

            if y.shape[0] != x.shape[0]:
                raise ValueError('The targets and features time series must have the same length.')

        if forecast_period < 1:
            raise ValueError('The length of the forecast period should be greater than or equal to one.')

        if lookback_period < forecast_period:
            raise ValueError('The length of the lookback period must be greater than the length of the forecast period.')

        if forecast_period + lookback_period >= y.shape[0]:
            raise ValueError('The combined length of the forecast and lookback periods must be less than the length of the time series.')

        if type(dilation_rates) != list:
            raise ValueError('The dilation rates must be provided as a list.')

        elif len(dilation_rates) == 0:
            dilation_rates = [1, 2, 4, 8]
            warnings.warn('The dilation rates were not provided, using [1, 2, 4, 8].')

        if type(quantiles) == list:
            q = np.array(quantiles)
        else:
            raise ValueError('The quantiles must be provided as a list.')

        if len(q) == 0:
            quantiles = [0.1, 0.5, 0.9]
            warnings.warn('The quantiles were not provided, using [0.1, 0.5, 0.9].')

        if 0.5 not in quantiles:
            quantiles = np.sort(np.append(0.5, quantiles))

        # Normalize the targets.
        y_min, y_max = np.min(y, axis=0), np.max(y, axis=0)
        y = (y - y_min) / (y_max - y_min)
        self.y_min = y_min
        self.y_max = y_max

        # Normalize the features.
        if x is not None:
            x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
            x = (x - x_min) / (x_max - x_min)
            self.x_min = x_min
            self.x_max = x_max

        # Save the inputs.
        self.y = y
        self.x = x
        self.q = np.array(quantiles)
        self.loss = loss
        self.n_outputs = 2 if loss == 'parameteric' else len(self.q)
        self.n_features = x.shape[1] if x is not None else 0
        self.n_samples = y.shape[0]
        self.n_targets = y.shape[1]
        self.n_quantiles = len(self.q)
        self.n_lookback = lookback_period
        self.n_forecast = forecast_period

        if x is not None:

            # Extract the input and output sequences.
            self.x_encoder, self.x_decoder, self.y_encoder, self.y_decoder = get_training_sequences_with_covariates(
                y=y,
                x=x,
                n_samples=self.n_samples,
                n_targets=self.n_targets,
                n_features=self.n_features,
                n_lookback=self.n_lookback,
                n_forecast=self.n_forecast
            )

            # Build the model graph.
            self.model = build_fn_with_covariates(
                n_targets=self.n_targets,
                n_features=self.n_features,
                n_outputs=self.n_outputs,
                n_lookback=self.n_lookback,
                n_forecast=self.n_forecast,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rates=dilation_rates,
                units=units,
                loss=self.loss
            )

        else:

            # Extract the input and output sequences.
            self.y_encoder, self.y_decoder = get_training_sequences(
                y=y,
                n_samples=self.n_samples,
                n_targets=self.n_targets,
                n_lookback=self.n_lookback,
                n_forecast=self.n_forecast
            )

            # Build the model graph.
            self.model = build_fn(
                n_targets=self.n_targets,
                n_outputs=self.n_outputs,
                n_lookback=self.n_lookback,
                n_forecast=self.n_forecast,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rates=dilation_rates,
                loss=self.loss
            )

    def fit(self,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            validation_split=0):

        '''
        Train the model.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        validation_split: float.
            Fraction of the training data to be used as validation data, must be between 0 and 1.
        '''

        # Compile the model.
        if self.loss == 'parametric':

            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=parametric_loss,
            )

        else:
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=lambda y_true, y_pred: nonparametric_loss(y_true, y_pred, self.q)
            )

        # Fit the model.
        if self.x is not None:

            self.model.fit(
                x=[self.x_encoder, self.x_decoder, self.y_encoder],
                y=self.y_decoder,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split
            )

        else:

            self.model.fit(
                x=self.y_encoder,
                y=self.y_decoder,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split
            )

    def predict(self, index):

        '''
        Extract the in-sample predictions.

        Parameters:
        __________________________________
        index: int.
            The start index of the sequence to predict.

        Returns:
        __________________________________
        predictions: pd.DataFrame
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        if index < self.n_lookback:
            raise ValueError('The index must be greater than {}.'.format(self.n_lookback))

        elif index > len(self.y) - self.n_forecast:
            raise ValueError('The index must be less than {}.'.format(self.n_samples - self.n_forecast))

        # Extract the predictions for the selected sequence.
        if self.x is not None:
            y_pred = self.model.predict([self.x_encoder, self.x_decoder, self.y_encoder])
        else:
            y_pred = self.model.predict(self.y_encoder)

        y_pred = y_pred[index - self.n_lookback: index - self.n_lookback + 1, :, :, :]

        # Organize the predictions in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.n_targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.q[j]) for i in range(self.n_targets) for j in range(self.n_quantiles)])

        predictions = pd.DataFrame(columns=columns)
        predictions['time_idx'] = np.arange(self.n_samples)

        for i in range(self.n_targets):
            predictions['target_' + str(i + 1)] = self.y_min[i] + (self.y_max[i] - self.y_min[i]) * self.y[:, i]
            for j in range(self.n_quantiles):
                if self.loss == 'parametric':
                    predictions['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[index: index + self.n_forecast] = \
                    self.y_min[i] + (self.y_max[i] - self.y_min[i]) * norm_ppf(y_pred[:, :, i, 0], y_pred[:, :, i, 1], self.q[j])
                else:
                    predictions['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[index: index + self.n_forecast] = \
                    self.y_min[i] + (self.y_max[i] - self.y_min[i]) * y_pred[:, :, i, j].flatten()

        predictions = predictions.astype(float)

        # Save the data frame.
        self.predictions = predictions

        # Return the data frame.
        return predictions

    def forecast(self, x=None):

        '''
        Generate the out-of-sample forecasts.

        Parameters:
        __________________________________
        x: np.array.
            Features time series, array with shape (n_forecast, n_features) where n_forecast is the length
            of the forecast period (decoder length) and n_features is the number of features time series.

        Returns:
        __________________________________
        forecasts: pd.DataFrame
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        if x is not None:

            if type(x) != np.ndarray:
                raise ValueError('The features time series must be provided as a numpy array.')

            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=1)

            elif len(x.shape) > 2:
                raise ValueError('The features array cannot have more than 2 dimensions. Found {} dimensions.'.format(len(x.shape)))

            if x.shape[0] != self.n_forecast:
                raise ValueError('The length of the features time series must be equal to the length of the forecast period.')

        # Generate the forecasts.
        y_encoder = self.y[- self.n_lookback:, :].reshape(1, self.n_lookback, self.n_targets)

        if self.x is not None:
            x_encoder = self.x[- self.n_lookback:, :].reshape(1, self.n_lookback, self.n_features)
            x_decoder = (x - self.x_min) / (self.x_max - self.x_min)
            x_decoder = x_decoder.reshape(1, x_decoder.shape[0], x_decoder.shape[1])
            y_pred = self.model.predict([x_encoder, x_decoder, y_encoder])
        else:
            y_pred = self.model.predict(y_encoder)

        # Organize the forecasts in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.n_targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.q[j]) for i in range(self.n_targets) for j in range(self.n_quantiles)])

        forecasts = pd.DataFrame(columns=columns)
        forecasts['time_idx'] = np.arange(self.n_samples + self.n_forecast)

        for i in range(self.n_targets):
            forecasts['target_' + str(i + 1)].iloc[: - self.n_forecast] = \
                self.y_min[i] + (self.y_max[i] - self.y_min[i]) * self.y[:, i]
            for j in range(self.n_quantiles):
                if self.loss == 'parametric':
                    forecasts['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[- self.n_forecast:] = \
                    self.y_min[i] + (self.y_max[i] - self.y_min[i]) * norm_ppf(y_pred[:, :, i, 0], y_pred[:, :, i, 1], self.q[j])
                else:
                    forecasts['target_' + str(i + 1) + '_' + str(self.q[j])].iloc[- self.n_forecast:] = \
                    self.y_min[i] + (self.y_max[i] - self.y_min[i]) * y_pred[:, :, i, j].flatten()

        forecasts = forecasts.astype(float)

        # Save the data frame.
        self.forecasts = forecasts

        # Return the data frame.
        return forecasts

    def plot_predictions(self):

        '''
        Plot the in-sample predictions.

        Returns:
        __________________________________
        go.Figure
        '''

        return plot(self.predictions, self.q, self.n_targets, self.n_quantiles)

    def plot_forecasts(self):

        '''
        Plot the out-of-sample forecasts.

        Returns:
        __________________________________
        go.Figure
        '''

        return plot(self.forecasts, self.q, self.n_targets, self.n_quantiles)


def build_fn_with_covariates(
        n_targets,
        n_features,
        n_outputs,
        n_lookback,
        n_forecast,
        filters,
        kernel_size,
        dilation_rates,
        units,
        loss):

    '''
    Build the model graph with covariates.

    Parameters:
    __________________________________
    n_targets: int
        Number of target time series.

    n_features: int.
        Number of features time series.

    n_outputs: int.
        Number of outputs, equal to 2 when the loss is parametric (in which case the two outputs are the mean
        and standard deviation of the Normal distribution), equal to the number of quantiles when the loss is
        nonparametric.

    n_lookback: int
        Length of input sequences.

    n_forecast: int
        Length of output sequences.

    filters: int.
        Number of filters (or channels) of the convolutional layers in the encoder module.

    kernel_size: int.
        Kernel size of the convolutional layers in the encoder module.

    dilation_rate: int.
        Dilation rates of the convolutional layers in the encoder module.

    units: int.
        Hidden units of dense layers in the decoder module.

    loss: str.
        The loss function, either 'nonparametric' or 'parametric'.
    '''

    # Define the inputs.
    x_encoder = Input(shape=(n_lookback, n_features))
    x_decoder = Input(shape=(n_forecast, n_features))
    y_encoder = Input(shape=(n_lookback, n_targets))

    # Concatenate the encoder inputs.
    encoder_input = concatenate([x_encoder, y_encoder], axis=2)

    # Forward pass the encoder inputs through the encoder module.
    for i in range(len(dilation_rates)):

        if i == 0:

            encoder_output = encoder(
                encoder_input=encoder_input,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i])

        else:

            encoder_output = encoder(
                encoder_input=encoder_output,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i])

    # Slice the second dimension of the encoder output to match the second dimension of the decoder input.
    encoder_output = Lambda(function=lambda x: x[:, - n_forecast:, :])(encoder_output)

    # Forward pass the decoder inputs and the sliced encoder output through the decoder module.
    decoder_ouput = decoder(
        decoder_input=x_decoder,
        encoder_output=encoder_output,
        units=units)

    # Forward pass the decoder outputs through the dense layer.
    decoder_ouput = Dense(units=n_targets * n_outputs)(decoder_ouput)

    # Reshape the decoder output to match the shape required by the loss function.
    y_decoder = Reshape(target_shape=(n_forecast, n_targets, n_outputs))(decoder_ouput)

    # If using the parametric loss, apply the soft ReLU activation to ensure a positive standard deviation.
    if loss == 'parametric':
        y_decoder = Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(y_decoder)

    return Model([x_encoder, x_decoder, y_encoder], y_decoder)


def build_fn(
        n_targets,
        n_outputs,
        n_lookback,
        n_forecast,
        filters,
        kernel_size,
        dilation_rates,
        loss):

    '''
    Build the model graph without covariates.

    Parameters:
    __________________________________
    n_targets: int
        Number of target time series.

    n_outputs: int.
        Number of outputs, equal to 2 when the loss is parametric (in which case the two outputs are the mean
        and standard deviation of the Normal distribution), equal to the number of quantiles when the loss is
        nonparametric.

    n_lookback: int
        Length of input sequences.

    n_forecast: int
        Length of output sequences.

    filters: int.
        Number of filters (or channels) of the convolutional layers in the encoder module.

    kernel_size: int.
        Kernel size of the convolutional layers in the encoder module.

    dilation_rate: int.
        Dilation rates of the convolutional layers in the encoder module.

    loss: str.
        The loss function, either 'nonparametric' or 'parametric'.
    '''

    # Define the inputs.
    y_encoder = Input(shape=(n_lookback, n_targets))

    # Forward pass the encoder inputs through the encoder module.
    for i in range(len(dilation_rates)):

        if i == 0:

            encoder_output = encoder(
                encoder_input=y_encoder,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i])

        else:

            encoder_output = encoder(
                encoder_input=encoder_output,
                filters=filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rates[i])

    # Slice the second dimension of the encoder output to match the output dimension.
    encoder_output = Lambda(function=lambda x: x[:, - n_forecast:, :])(encoder_output)

    # Forward pass the encoder outputs through the dense layer.
    encoder_output = Dense(units=n_targets * n_outputs)(encoder_output)

    # Reshape the encoder output to match the shape required by the loss function.
    y_decoder = Reshape(target_shape=(n_forecast, n_targets, n_outputs))(encoder_output)

    # If using the parametric loss, apply the soft ReLU activation to ensure a positive standard deviation.
    if loss == 'parametric':
        y_decoder = Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(y_decoder)

    return Model(y_encoder, y_decoder)


def soft_relu(x):
    '''
    Soft ReLU activation function, used for ensuring the positivity of the standard deviation of the Normal distribution
    when using the parameteric loss function. See Section 3.2.2 in the DeepTCN paper.
    '''

    return tf.math.log(1.0 + tf.math.exp(x))

def norm_ppf(loc, scale, value):

    '''
    Quantiles of the Normal distribution.
    '''

    return tfp.distributions.Normal(loc, scale).quantile(value).numpy().flatten()


