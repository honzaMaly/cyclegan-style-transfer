import tensorflow as tf
from matplotlib import pyplot as plt

# default initializer for weights
default_initializer = tf.random_normal_initializer(0., 0.02)


class LayerConfiguration:
    """
    Abstract class to represent a configuration for single layer
    """

    def __init__(self, filters, kernel_size=3):
        self.filters = filters
        self.kernel_size = kernel_size


class DownsampleLayerConfiguration(LayerConfiguration):
    """
    Class to represent configuration for downsampling layer
    """

    def __init__(self, filters, kernel_size=3, apply_norm=True):
        super().__init__(filters, kernel_size)
        self.apply_norm = apply_norm


class UNetLayerConfiguration(DownsampleLayerConfiguration):
    """
    Class to represent configuration for pair of connected layers - downsampling (in the encoder) and
    upsampling (in the decoder) in a network with the UNet architecture
    """

    def __init__(self, filters, kernel_size=3, apply_norm=True, dropout=0.0):
        super().__init__(filters, kernel_size, apply_norm)
        self.dropout = dropout


# define default architecture of generator based on: https://www.tensorflow.org/tutorials/generative/cyclegan
u_net_encoder_decoder_layers = tuple([UNetLayerConfiguration(64, 4, apply_norm=False),
                                      UNetLayerConfiguration(128, 4),
                                      UNetLayerConfiguration(256, 4),
                                      UNetLayerConfiguration(512, 4),
                                      UNetLayerConfiguration(512, 4, dropout=0.5),
                                      UNetLayerConfiguration(512, 4, dropout=0.5),
                                      UNetLayerConfiguration(512, 4, dropout=0.5)])
u_net_connecting_layers = tuple([DownsampleLayerConfiguration(512, 4)])

# define default architecture of discriminator based on: https://www.tensorflow.org/tutorials/generative/cyclegan
discriminator_downsampling_layers = tuple([DownsampleLayerConfiguration(64, 4, apply_norm=False),
                                           DownsampleLayerConfiguration(128, 4),
                                           DownsampleLayerConfiguration(256, 4)])
discriminator_final_convolution_layer = LayerConfiguration(512, 4)


class InstanceNormalization(tf.keras.layers.Layer):
    """
    Instance Normalization Layer: https://arxiv.org/abs/1607.08022
    """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = None
        self.offset = None

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(1., 0.02)
        self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer=initializer, trainable=True)
        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def downsample(filters, kernel_size, apply_norm):
    """
    Downsamples an input.
    Conv2D => Instancenorm => LeakyRelu
    :param filters: number of filters
    :param kernel_size: filter size
    :param apply_norm: If True, adds the instancenorm layer
    :return: downsample Sequential Model
    """
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                                      kernel_initializer=default_initializer, use_bias=False))
    if apply_norm:
        result.add(InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, kernel_size, dropout):
    """
    Upsamples an input.
    Conv2DTranspose => Instancenorm => Dropout => Relu
    :param filters: number of filters
    :param kernel_size: filter size
    :param dropout: dropout to apply - number in [0, 1)
    :return: Upsample Sequential Model
    """
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same',
                                               kernel_initializer=default_initializer, use_bias=False))
    result.add(InstanceNormalization())
    if 0 < dropout < 1:
        result.add(tf.keras.layers.Dropout(dropout))
    result.add(tf.keras.layers.ReLU())
    return result


def u_net_generator(output_channels=3, input_shape=(256, 256, 3), encoder_decoder_layers=u_net_encoder_decoder_layers,
                    connecting_layers=u_net_connecting_layers, kernel_size_last_layer=4):
    """
    Modified u-net generator model: https://arxiv.org/abs/1611.07004
    :param kernel_size_last_layer: filter size for last layer
    :param encoder_decoder_layers: configuration for connected encoding/decoding layers
    :param connecting_layers: configuration for layers connecting encoder with decoder
    :param output_channels: Output channels
    :param input_shape: shape of input data
    :return: Generator model
    """

    # create input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # layers for the encoder part of the model
    # downsampling through the model
    skips = []
    for layer_configuration in encoder_decoder_layers:
        down = downsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.apply_norm)
        x = down(x)
        skips.append(x)

    # add inter-layer
    for layer_configuration in connecting_layers:
        down = downsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.apply_norm)
        x = down(x)

    # layers for the decoder part of the model
    # upsampling and establishing the skip connections
    for i in reversed(range(len(encoder_decoder_layers))):
        layer_configuration = encoder_decoder_layers[i]
        up = upsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.dropout)
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skips[i]])

    # create last layer
    last = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size_last_layer, strides=2, padding='same',
                                           kernel_initializer=default_initializer, activation='tanh')
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(input_shape=(256, 256, 3), downsampling_layers=discriminator_downsampling_layers,
                  final_convolution_layer=discriminator_final_convolution_layer, kernel_size_last_layer=3):
    """
    PatchGan discriminator model: https://arxiv.org/abs/1611.07004
    :param final_convolution_layer: configuration for final convolution layer
    :param downsampling_layers: configuration for downsampling layers
    :param input_shape: shape of input data
    :param kernel_size_last_layer: filter size for last layer
    :return: Discriminator model
    """

    # define input layer
    inp = tf.keras.layers.Input(shape=input_shape, name='input_image')

    # stack layers
    x = inp
    for layer_configuration in downsampling_layers:
        down = downsample(layer_configuration.filters, layer_configuration.kernel_size, layer_configuration.apply_norm)
        x = down(x)

    # rest of the network before final layer (flatten channels)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(final_convolution_layer.filters, final_convolution_layer.kernel_size, strides=1,
                               kernel_initializer=default_initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)

    # add last layer
    last = tf.keras.layers.Conv2D(1, kernel_size_last_layer, strides=1, kernel_initializer=default_initializer)(x)

    return tf.keras.Model(inputs=inp, outputs=last)


def plot_transformations(image_sample_x, image_sample_y, generator_x2y, generator_y2x, distributions_names=('X', 'Y'),
                         fig_size=(8, 8), contrast=8):
    """
    Plots provided original samples from X and Y. Generates samples x->y and y->x given original samples and generators
    to plot them.
    :param image_sample_x: samples from X
    :param image_sample_y: samples from Y
    :param generator_x2y: generator mapping X->Y
    :param generator_y2x: generator mapping Y->X
    :param distributions_names: tuple of names for distribution X and Y
    :param fig_size: size of figure
    :param contrast: contrast to use when plotting image
    """

    plt.figure(figsize=fig_size)

    # plot single row
    def plot_row(image_sample, generator, row_index):
        # plot the original
        plt.subplot(2, 2, (row_index + 1) + row_index)
        plt.title("Original sample from '{}'".format(distributions_names[row_index % 2]))
        plt.imshow(image_sample * 0.5 * contrast + 0.5)

        # generate image from original and plot it
        plt.subplot(2, 2, (row_index + 1) * 2)
        plt.title("Converted sample from '{}' to '{}'"
                  .format(distributions_names[row_index % 2], distributions_names[(row_index + 1) % 2]))
        generated_img = generator(tf.cast([image_sample], tf.float32))[0]
        plt.imshow(generated_img * 0.5 * contrast + 0.5)

    # plot data
    plot_row(image_sample_x, generator_x2y, 0)
    plot_row(image_sample_y, generator_y2x, 1)
    plt.show()
