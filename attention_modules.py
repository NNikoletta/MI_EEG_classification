"""
this python file contains the attention modules

CBAM -> proposed by Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon
        in CBAM: Convolutional Block Attention Module (2018)
"""

from tensorflow import reduce_mean, expand_dims, reduce_max
from keras.layers import Dense, Activation, Concatenate, Multiply, Reshape
from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D


def ChannelAttentionEEG(input, ratio=8):  # x  = input feature map
    """
    Same architecture as the Channel Attention module of the CBAM but the dimensions are changed so that
    the function chooses the channels of the EEG signals
    :param input: input data
    :param ratio: reduction ratio
    :return: input with weighted channels
    """
    bm, channels, timesteps, filters = input.shape
    # Shared layers
    l1 = Dense(channels//ratio, activation='relu', use_bias=False)
    l2 = Dense(channels, use_bias=False)

    # Global Average Pooling
    x1 = GlobalAveragePooling2D(data_format='channels_first')(input)
    x1 = l1(x1)
    x1 = l2(x1)

    # Global Max Pool
    x2 = GlobalMaxPooling2D(data_format='channels_first')(input)
    x2 = l1(x2)
    x2 = l2(x2)

    out = x1 + x2

    out = Activation("sigmoid")(out)
    input_reshape = Reshape((filters, timesteps, channels))(input)

    out = Multiply()([input_reshape, out])
    out = Reshape((channels, timesteps, filters))(out)

    return out


def TemporalAttention(input, kernel_size=7):  # input  = input feature map
    """
    Same architecture as the Spatial Attention of the CBAM module, but the dimensions are changed, so that the function
    chooses the timesteps of interest
    :param input: input EEG signals
    :param kernel_size: size of the Conv2D layer's kernel
    :return: weighted input
    """
    # Average Pooling
    x1 = reduce_mean(input, axis=1)
    x1 = expand_dims(x1, axis=1)

    # Max Pooling
    x2 = reduce_max(input, axis=1)
    x2 = expand_dims(x2, axis=1)

    # Concatenate
    out = Concatenate()([x1, x2])

    # Conv layer
    out = Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(out)
    out = Multiply()([input, out])

    return out


def AttentionModuleCTAM(x, ratio=16, kernel_size=7):
    x = ChannelAttentionEEG(x, ratio)
    x = TemporalAttention(x, kernel_size)
    return x


# --------------------------------------------------------------------------------------------------------------------

def FiltersAttention(input, ratio=16):  # x  = input feature map CHANNEL ATTENTION OF A CBAM
    """
    The Channel Attention Module of the CBAM, the name has been changed to FiltersAttention, because the dimensions of
    the EEG signals are not the same as the dimensions of an image. In this study, the last dimension, is the dimension
    of the filters
    :param input: input EEG signals
    :param ratio: reduction ratio
    :return: weighted input
    """
    bm, _, _, filters = input.shape
    # Shared layers
    l1 = Dense(filters//ratio, activation='relu', use_bias=False)
    l2 = Dense(filters, use_bias=False)

    # Global Average Pooling
    x1 = GlobalAveragePooling2D(data_format='channels_last')(input)
    x1 = l1(x1)
    x1 = l2(x1)

    # Global Max Pool
    x2 = GlobalMaxPooling2D(data_format='channels_last')(input)
    x2 = l1(x2)
    x2 = l2(x2)

    out = x1 + x2
    out = Activation("sigmoid")(out)
    out = Multiply()([input, out])

    return out


def SpatialAttention(input, kernel_size=7):  # x  = input feature map SPATIAL ATTENTION OF A CBAM
    """
    The Spatial Attention Module of the CBAM
    :param input: EEG signals
    :param kernel_size: size of the Conv2D layer's kernel
    :return: weighted input
    """
    # Average Pooling
    x1 = reduce_mean(input, axis=-1)
    x1 = expand_dims(x1, axis=-1)

    # Max Pooling
    x2 = reduce_max(input, axis=-1)
    x2 = expand_dims(x2, axis=-1)

    # Concatenate
    out = Concatenate()([x1, x2])

    # Conv layer
    out = Conv2D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(out)
    out = Multiply()([input, out])

    return out


def cbam(x, ratio=16, kernel_size=7):
    x = FiltersAttention(x, ratio)
    x = SpatialAttention(x, kernel_size)
    return x