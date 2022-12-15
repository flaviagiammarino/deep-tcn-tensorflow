import tensorflow as tf

def encoder(encoder_input, filters, kernel_size, dilation_rate):

    '''
    Encoder module, see Section 3.1.1 in the DeepTCN paper.

    Parameters:
    __________________________________
    encoder_input: tf.Tensor.
        For the first stack, this is a tensor with shape (n_samples, n_lookback, n_features + n_targets) where
        n_samples is the batch size, n_lookback is the encoder length, n_features is the number of features and
        n_targets is the number of targets. For the subsequent stacks, this is a tensor with shape (n_samples,
        n_lookback, filters) where filters is the number of channels of the convolutional layers.

    filters: int.
        Number of filters (or channels) of the convolutional layers.

    kernel_size: int.
        Kernel size of the convolutional layers.

    dilation_rate: int.
        Dilation rate of the convolutional layers.

    Returns:
    __________________________________
    encoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_lookback, filters) where n_samples is the batch size, n_lookback is
        the encoder length and filters is the number of channels of the convolutional layers.
    '''

    encoder_output = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(encoder_input)
    encoder_output = tf.keras.layers.BatchNormalization()(encoder_output)
    encoder_output = tf.keras.layers.ReLU()(encoder_output)
    encoder_output = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(encoder_output)
    encoder_output = tf.keras.layers.BatchNormalization()(encoder_output)

    if encoder_input.shape[-1] != encoder_output.shape[-1]:
        encoder_input = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(encoder_input)

    encoder_output = tf.keras.layers.Add()([encoder_input, encoder_output])
    encoder_output = tf.keras.layers.ReLU()(encoder_output)

    return encoder_output


def decoder(decoder_input, encoder_output, units):

    '''
    Decoder module, see Section 3.1.2 in the DeepTCN paper.

    Parameters:
    __________________________________
    decoder_input: tf.Tensor.
        A tensor with shape (n_samples, n_forecast, n_features) where n_samples is the batch size, n_forecast
        is the decoder length and n_features is the number of features.

    encoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_forecast, filters) where n_samples is the batch size, n_forecast
        is the decoder length and filters is the number of channels of the convolutional layers. Note that
        this is obtained by slicing the second dimension of the output of the encoder module to keep only
        the last n_forecast timesteps.

    units: int.
        The number of hidden units of the dense layers.

    Returns:
    __________________________________
    decoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_forecast, units) where n_samples is the batch size, n_forecast
        is the decoder length and units is the number of hidden units of the dense layers.
    '''

    decoder_output = tf.keras.layers.Dense(units=units)(decoder_input)
    decoder_output = tf.keras.layers.BatchNormalization()(decoder_output)
    decoder_output = tf.keras.layers.ReLU()(decoder_output)
    decoder_output = tf.keras.layers.Dense(units=units)(decoder_output)
    decoder_output = tf.keras.layers.BatchNormalization()(decoder_output)

    if encoder_output.shape[-1] != decoder_output.shape[-1]:
        encoder_output = tf.keras.layers.Dense(units=units)(encoder_output)

    decoder_output = tf.keras.layers.Add()([decoder_output, encoder_output])
    decoder_output = tf.keras.layers.ReLU()(decoder_output)

    return decoder_output
