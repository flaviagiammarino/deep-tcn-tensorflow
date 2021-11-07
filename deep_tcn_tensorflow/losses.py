import tensorflow as tf
import numpy as np

def nonparametric_loss(y_true, y_pred, q):

    '''
    Nonparametric loss function, see Section 3.2.1 in the DeepTCN paper.

    Parameters:
    __________________________________
    y_true: tf.Tensor.
        Actual values of target time series, a tensor with shape (n_samples, n_forecast, n_targets) where n_samples is
        the batch size, n_forecast is the length of output sequences and n_targets is the number of target time series.

    y_pred: tf.Tensor.
        Predicted quantiles of target time series, a tensor with shape (n_samples, n_forecast, n_targets, n_quantiles)
        where n_samples is the batch size, n_forecast is the length of output sequences, n_targets is the number of
        target time series and n_quantiles is the number of quantiles.

    q: tf.Tensor.
        Quantiles, a tensor with shape equal to the number of quantiles.

    Returns:
    __________________________________
    tf.Tensor.
        Loss value, a tensor with shape 1.
    '''

    y_true = tf.cast(tf.expand_dims(y_true, axis=3), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    q = tf.cast(tf.reshape(q, shape=(1, len(q))), dtype=tf.float32)
    e = tf.subtract(y_true, y_pred)

    L = tf.multiply(q, tf.maximum(0.0, e)) + tf.multiply(1.0 - q, tf.maximum(0.0, - e))

    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(L, axis=-1), axis=-1))


def parametric_loss(y_true, params):

    '''
    Parametric loss function, see Section 3.2.2 in the DeepTCN paper.

    Parameters:
    __________________________________
    y_true: tf.Tensor.
        Actual values of target time series, a tensor with shape (n_samples, n_forecast, n_targets) where n_samples is
        the batch size, n_forecast is the length of output sequences and n_targets is the number of target time series.

    params: tf.Tensor.
        Predicted means and standard deviations of target time series, a tensor with shape (n_samples, n_forecast,
        n_targets, 2) where n_samples is the batch size, n_forecast is the length of output sequences and n_targets
        is the number of target time series.

    Returns:
    __________________________________
    tf.Tensor.
        Loss value, a tensor with shape 1.
    '''

    y_true = tf.cast(y_true, dtype=tf.float32)
    params = tf.cast(params, dtype=tf.float32)

    mu = params[:, :, :, 0]
    sigma = params[:, :, :, 1]

    L = 0.5 * tf.math.log(2 * np.pi) + tf.math.log(sigma) + tf.math.divide(tf.math.pow(y_true - mu, 2), 2 * tf.math.pow(sigma, 2))

    return tf.experimental.numpy.nanmean(tf.experimental.numpy.nanmean(tf.experimental.numpy.nansum(L, axis=-1), axis=-1))
