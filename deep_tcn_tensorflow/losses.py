import tensorflow as tf

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

    L = tf.multiply(q, tf.maximum(0.0, tf.subtract(y_true, y_pred))) + tf.multiply(1.0 - q, tf.maximum(0.0, - tf.subtract(y_true, y_pred)))

    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(L, axis=-1), axis=-1))
