import tensorflow as tf
import tensorlayer as tl

'''
This is where we define our own forward transform network for mask recovery
'''
def conv_layers(net_in):

    # network = tl.layers.BatchNormLayer(net_in,
    #                                    decay=0.999,
    #                                    epsilon=1e-05,
    #                                    is_train=True,
    #                                    name='norm1')

    #""" conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                    shape = [1, 1, 2, 16],  # 64 features for each 9x9 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv1')

    network = tl.layers.Conv2dLayer(network,
                    shape = [3, 3, 16, 8],  # 64 features for each 9x9 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv2')

    network = tl.layers.BatchNormLayer(network,
                                       decay=0.999,
                                       epsilon=1e-05,
                                       is_train=True,
                                       name='norm1')

    # """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [11, 11, 8, 32],  # 32 features for each 1x1 patch
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv3')

    network = tl.layers.BatchNormLayer(network,
                                       decay=0.999,
                                       epsilon=1e-05,
                                       is_train=True,
                                       name='norm2')

    # """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tf.nn.relu,
                    shape = [1, 1, 32, 16],  # filter of size 5x5, resize features to have the output [1,96,96,1]
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv4')

    network = tl.layers.BatchNormLayer(network,
                                       decay=0.999,
                                       epsilon=1e-05,
                                       is_train=True,
                                       name='norm3')

    # """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                    act = tl.activation.ramp,
                    shape = [5, 5, 16, 1],  # filter of size 5x5, resize features to have the output [1,96,96,1]
                    strides = [1, 1, 1, 1],
                    padding='SAME',
                    name ='conv5')

    net_out = network

    return net_out