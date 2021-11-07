import numpy as np

def get_training_sequences_with_covariates(y, x, n_samples, n_targets, n_features, n_lookback, n_forecast):

    '''
    Split the time series into input and output sequences. These are used for training the model.
    See Section 3 in the DeepTCN paper.

    Parameters:
    __________________________________
    y: np.array
        Target time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
        and n_targets is the number of target time series.

    x: np.array
        Features time series, array with shape (n_samples, n_features) where n_samples is the length of the time series
        and n_features is the number of features time series.

    n_samples: int
        Length of the time series.

    n_targets: int
        Number of target time series.

    n_features: int.
        Number of features time series.

    n_lookback: int
        Length of input sequences.

    n_forecast: int
        Length of output sequences.

    Returns:
    __________________________________
    x_encoder: np.array.
        Encoder features, array of with shape (n_samples - n_lookback - n_forecast + 1, n_lookback, n_features).

    x_decoder: np.array.
        Decoder features, array of with shape (n_samples - n_lookback - n_forecast + 1, n_forecast, n_features).

    y_encoder: np.array.
        Encoder targets, array of with shape (n_samples - n_lookback - n_forecast + 1, n_lookback, n_targets).

    y_decoder: np.array.
        Decoder targets, array of with shape (n_samples - n_lookback - n_forecast + 1, n_forecast, n_targets).
    '''

    x_encoder = np.zeros((n_samples, n_lookback, n_features))
    x_decoder = np.zeros((n_samples, n_forecast, n_features))

    y_encoder = np.zeros((n_samples, n_lookback, n_targets))
    y_decoder = np.zeros((n_samples, n_forecast, n_targets))

    for i in range(n_lookback, n_samples - n_forecast + 1):

        x_encoder[i, :, :] = x[i - n_lookback: i]
        x_decoder[i, :, :] = x[i: i + n_forecast]

        y_encoder[i, :, :] = y[i - n_lookback: i]
        y_decoder[i, :, :] = y[i: i + n_forecast]

    x_encoder = x_encoder[n_lookback: n_samples - n_forecast + 1, :, :]
    x_decoder = x_decoder[n_lookback: n_samples - n_forecast + 1, :, :]

    y_encoder = y_encoder[n_lookback: n_samples - n_forecast + 1, :, :]
    y_decoder = y_decoder[n_lookback: n_samples - n_forecast + 1, :, :]

    return x_encoder, x_decoder, y_encoder, y_decoder


def get_training_sequences(y, n_samples, n_targets, n_lookback, n_forecast):

    '''
    Split the time series into input and output sequences. These are used for training the model.
    See Section 3 in the DeepTCN paper.

    Parameters:
    __________________________________
    y: np.array
        Target time series, array with shape (n_samples, n_targets) where n_samples is the length of the time series
        and n_targets is the number of target time series.

    n_samples: int
        Length of the time series.

    n_targets: int
        Number of target time series.

    n_lookback: int
        Length of input sequences.

    n_forecast: int
        Length of output sequences.

    Returns:
    __________________________________
    y_encoder: np.array.
        Encoder targets, array of with shape (n_samples - n_lookback - n_forecast + 1, n_lookback, n_targets).

    y_decoder: np.array.
        Decoder targets, array of with shape (n_samples - n_lookback - n_forecast + 1, n_forecast, n_targets).
    '''

    y_encoder = np.zeros((n_samples, n_lookback, n_targets))
    y_decoder = np.zeros((n_samples, n_forecast, n_targets))

    for i in range(n_lookback, n_samples - n_forecast + 1):

        y_encoder[i, :, :] = y[i - n_lookback: i]
        y_decoder[i, :, :] = y[i: i + n_forecast]

    y_encoder = y_encoder[n_lookback: n_samples - n_forecast + 1, :, :]
    y_decoder = y_decoder[n_lookback: n_samples - n_forecast + 1, :, :]

    return y_encoder, y_decoder


