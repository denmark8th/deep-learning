import tensorflow as tf
import numpy as np
import scipy.io
from scipy import misc


def net(data_path, input_image):
    # Before applying VGG network, the mean value should be subtracted from the image data
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(data_path)
    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels = weights[i][0][0][2][0][0]
            bias = weights[i][0][0][2][0][1]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias, name=name)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = _pool_layer(current, name=name)
        net[name] = current

    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias, name=None):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME', name=name)
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, name=None):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME', name=name)


def preprocess(image):
    # Here image should be in the shape [batch_size,96,96,1]
    mean_pixel = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return image - mean_pixel


def unprocess(image, mean_pixel):
    # Here image should be in the shape [batch_size,96,96,1]
    mean_pixel = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return image + mean_pixel
