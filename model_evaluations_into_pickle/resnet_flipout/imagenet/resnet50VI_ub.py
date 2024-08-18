# Adapted from Uncertainty Baselines - https://github.com/google/uncertainty-baselines
# Base Resnet50 Variational Model - Edward2 Dependency
# Adapting author - John Fischer

import functools
import string
from absl import logging

import numpy as np
import tensorflow as tf

from variational_utils import get_kernel_regularizer_class
from variational_utils import init_kernel_regularizer

import edward2 as ed

BatchNormalization = functools.partial(tf.keras.layers.BatchNormalization, epsilon=1e-5, momentum=0.9)
Conv2DFlipout = functools.partial(ed.layers.Conv2DFlipout, use_bias=False)

def bottleneck_block(inputs,
                    filters,
                    stage,
                    block,
                    strides,
                    prior_stddev,
                    dataset_size,
                    stddev_mean_init,
                    stddev_stddev_init,
                    tied_mean_prior=True):
    
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    kernel_regularizer_class = get_kernel_regularizer_class(tied_mean_prior=tied_mean_prior)

    kernel_regularizer_2a = init_kernel_regularizer(
        kernel_regularizer_class,
        dataset_size,
        prior_stddev,
        inputs,
        n_filters=filters1,
        kernel_size=1)
    x = Conv2DFlipout(
        filters1,
        kernel_size=1,
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_mean_init)),
                stddev=stddev_stddev_init)),
        kernel_regularizer=kernel_regularizer_2a,
        name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    kernel_regularizer_2b = init_kernel_regularizer(
        kernel_regularizer_class,
        dataset_size,
        prior_stddev,
        x,
        n_filters=filters2,
        kernel_size=3)
    x = Conv2DFlipout(
        filters2,
        kernel_size=3,
        strides=strides,
        padding='same',
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_mean_init)),
                stddev=stddev_stddev_init)),
        kernel_regularizer=kernel_regularizer_2b,
        name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    kernel_regularizer_2c = init_kernel_regularizer(
        kernel_regularizer_class,
        dataset_size,
        prior_stddev,
        x,
        n_filters=filters3,
        kernel_size=1)
    x = Conv2DFlipout(
        filters3,
        kernel_size=1,
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_mean_init)),
                stddev=stddev_stddev_init)),
        kernel_regularizer=kernel_regularizer_2c,
        name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = inputs
    if not x.shape.is_compatible_with(shortcut.shape):
        kernel_regularizer_1 = init_kernel_regularizer(
            kernel_regularizer_class,
            dataset_size,
            prior_stddev,
            shortcut,
            n_filters=filters3,
            kernel_size=1)
        shortcut = Conv2DFlipout(
            filters3,
            kernel_size=1,
            strides=strides,
            kernel_initializer=ed.initializers.TrainableHeNormal(
                stddev_initializer=tf.keras.initializers.TruncatedNormal(
                    mean=np.log(np.expm1(stddev_mean_init)),
                    stddev=stddev_stddev_init)),
            kernel_regularizer=kernel_regularizer_1,
            name=conv_name_base + '1')(shortcut)
        shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def group(inputs, filters, num_blocks, stage, strides, prior_stddev, dataset_size, 
        stddev_mean_init, stddev_stddev_init):
  
    blocks = string.ascii_lowercase

    x = bottleneck_block(
        inputs,
        filters,
        stage,
        block=blocks[0],
        strides=strides,
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_mean_init=stddev_mean_init,
        stddev_stddev_init = stddev_stddev_init)
  
    for i in range(num_blocks - 1):
        x = bottleneck_block(
            x,
            filters,
            stage,
            block=blocks[i + 1],
            strides=1,
            prior_stddev=prior_stddev,
            dataset_size=dataset_size,
            stddev_mean_init=stddev_mean_init,
            stddev_stddev_init=stddev_stddev_init)
    
    return x


def resnet50_variational(input_shape, num_classes, prior_stddev, dataset_size, stddev_mean_init, 
                         stddev_stddev_init, tied_mean_prior=True, include_top=True):

    kernel_regularizer_class = get_kernel_regularizer_class(tied_mean_prior=tied_mean_prior)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)

    # Initialize kernel with given fixed stddev for prior, or compute the
    # stddev as sqrt(2 / fan_in) (as is done for the stddev in He initialization).
    kernel_regularizer_conv1 = init_kernel_regularizer(
        kernel_regularizer_class,
        dataset_size,
        prior_stddev,
        x,
        n_filters=64,
        kernel_size=7)
    x = Conv2DFlipout(
        64,
        kernel_size=7,
        strides=2,
        padding='valid',
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_mean_init)),
                stddev=stddev_stddev_init)),
        kernel_regularizer=kernel_regularizer_conv1,
        name='conv1')(x)
  
    x = BatchNormalization(name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(3,strides=2, padding='same')(x)

    x = group(
        x, [64, 64, 256],
        stage=2,
        num_blocks=3,
        strides=1,
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_mean_init=stddev_mean_init,
        stddev_stddev_init=stddev_stddev_init)
    x = group(
        x, 
        [128, 128, 512],
        stage=3,
        num_blocks=4,
        strides=2,
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_mean_init=stddev_mean_init,
        stddev_stddev_init=stddev_stddev_init)
    x = group(
        x, 
        [256, 256, 1024],
        stage=4,
        num_blocks=6,
        strides=2,
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_mean_init=stddev_mean_init,
        stddev_stddev_init=stddev_stddev_init)
    x = group(
        x, [512, 512, 2048],
        stage=5,
        num_blocks=3,
        strides=2,
        prior_stddev=prior_stddev,
        dataset_size=dataset_size,
        stddev_mean_init=stddev_mean_init,
        stddev_stddev_init=stddev_stddev_init)
  
    if not include_top:
        return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50_variational')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    kernel_regularizer_fc = init_kernel_regularizer(
        kernel_regularizer_class,
        dataset_size,
        prior_stddev,
        x,
        n_outputs=num_classes)
    x = ed.layers.DenseFlipout(
        num_classes,
        activation='softmax',
        kernel_initializer=ed.initializers.TrainableHeNormal(
            stddev_initializer=tf.keras.initializers.TruncatedNormal(
                mean=np.log(np.expm1(stddev_mean_init)),
                stddev=stddev_stddev_init)),
        kernel_regularizer=kernel_regularizer_fc,
        name='fc1000')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50_variational')
