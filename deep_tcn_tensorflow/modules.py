from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dense, Add

def encoder(encoder_input, filters, kernel_size, dilation_rate):

    '''
    Encoder module, see Section 3.1.1 in the DeepTCN paper.

    Parameters:
    __________________________________
    encoder_input: tf.Tensor.
        For the first stack, this is a tensor with shape (n_samples, n_lookback, n_features + n_targets)
        where n_samples is the batch size, n_lookback is the length of input sequences, n_features is the
        number of features and n_targets is the number of targets. For the subsequent stacks, this is a
        tensor with shape (n_samples, n_lookback, filters) where filters is the number of output filters
        (or channels) of the convolutional layers.

    filters: int.
        Number of filters (or channels) of the convolutional layers.

    kernel_size: int.
        Kernel size of the convolutional layers.

    dilation_rate: int.
        Dilation rates of the convolutional layers.

    Returns:
    __________________________________
    encoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_lookback, filters) where n_samples is the batch size, n_lookback
        is the length of input sequences and filters is the number of output filters (or channels) of the
        convolutional layers.
    '''

    encoder_output = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(encoder_input)
    encoder_output = BatchNormalization()(encoder_output)
    encoder_output = ReLU()(encoder_output)
    encoder_output = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='causal')(encoder_output)
    encoder_output = BatchNormalization()(encoder_output)

    # Adjust the width of the encoder input if it is different from the width of
    # the encoder output, see Section 3.4 in https://arxiv.org/abs/1803.01271.
    if encoder_input.shape[-1] != encoder_output.shape[-1]:
        encoder_input = Conv1D(filters=1, kernel_size=kernel_size, padding='causal')(encoder_input)

    encoder_output = Add()([encoder_input, encoder_output])
    encoder_output = ReLU()(encoder_output)

    return encoder_output


def decoder(decoder_input, encoder_output, units):

    '''
    Decoder module, see Section 3.1.2 in the DeepTCN paper.

    Parameters:
    __________________________________
    decoder_input: tf.Tensor.
        A tensor with shape (n_samples, n_forecast, n_features) where n_samples is the batch size, n_forecast
        is the length of output sequences and n_features is the number of features.

    encoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_forecast, filters) where n_samples is the batch size, n_forecast
        is the length of output sequences and filters is the number of output filters (or channels) of the
        convolutional layers. Note that this is obtained by slicing the second dimension of the output of
        the encoder module to keep only the last n_forecast timesteps.

    units: int.
        The number of hidden units of the dense layers.

    Returns:
    __________________________________
    decoder_output: tf.Tensor.
        A tensor with shape (n_samples, n_forecast, units) where n_samples is the batch size, n_forecast
        is the length of output sequences and units is the number of hidden units of the dense layers.
    '''

    decoder_output = Dense(units=units)(decoder_input)
    decoder_output = BatchNormalization()(decoder_output)
    decoder_output = ReLU()(decoder_output)
    decoder_output = Dense(units=units)(decoder_output)
    decoder_output = BatchNormalization()(decoder_output)

    # Adjust the width of the decoder input if it is different from the width of
    # the decoder output, see Section 3.4 in https://arxiv.org/abs/1803.01271.
    if encoder_output.shape[-1] != decoder_output.shape[-1]:
        encoder_output = Dense(units=1)(encoder_output)

    decoder_output = Add()([decoder_output, encoder_output])
    decoder_output = ReLU()(decoder_output)

    return decoder_output