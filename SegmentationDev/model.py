import tensorflow as tf
import numpy as np
import math

def xavier_init(n_inputs, n_outputs, uniform=True):
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

def weights_variables(shape, name):
    #init = tf.get_variable("W", shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    init = tf.truncated_normal(shape, stddev=0.1)

    #W = tf.get_variable(name, shape=shape,
    #                    initializer=tf.contrib.layers.xavier_initializer())

    """
    weights = tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.truncated_normal_initializer(
            stddev=0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.01)
    )
    """
    return tf.Variable(init, name = name)

def bias_variables(shape, name):
    init = tf.constant(0.1, shape = shape)
    #b = tf.get_variable(name, shape=shape,
    #                initializer=tf.contrib.layers.xavier_initializer())

    return tf.Variable(init, name=name)


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]], mode=mode)
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    with tf.variable_scope('conv_transpose') as scope:
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
